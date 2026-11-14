"""Validate and bundle the editable single-column paper and its figures."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "final/latex"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    texfiles = sorted(SOURCE.rglob("*.tex"))
    assert (SOURCE / "main.tex").is_file()
    text = "\n".join(p.read_text(encoding="utf-8-sig") for p in texfiles)
    # Ignore full-line comments when checking executable document structure.
    active = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))
    assert "twocolumn" not in active
    assert re.search(r"\\documentclass(?:\[[^]]*\])?\{article\}", active)
    assert not re.search(r"[\u4e00-\u9fff]", active)
    for name in re.findall(r"\\input\{([^}]+)\}", active):
        path = SOURCE / name
        assert path.with_suffix(".tex").is_file() if not path.suffix else path.is_file(), name
    images = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", active)
    assert len(images) >= 4
    for name in images:
        assert (SOURCE / name).is_file() or (SOURCE / "figures" / name).is_file(), name
    labels = re.findall(r"\\label\{([^}]+)\}", active)
    assert all(n == 1 for n in Counter(labels).values()), "Duplicate labels"
    refs = re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", active)
    assert set(refs) <= set(labels), set(refs) - set(labels)
    bib = (SOURCE / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cites = set()
    for group in re.findall(r"\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}", active):
        cites.update(key.strip() for key in group.split(","))
    assert cites <= keys, cites - keys
    allowed = {".tex", ".bib", ".pdf", ".png", ".md", ".txt", ".json"}
    files = [p for p in sorted(SOURCE.rglob("*")) if p.is_file() and p.suffix in allowed
             and p.name not in {"manifest.json", "main.pdf"}]
    manifest = {
        "format": "Single-column 11pt article; English",
        "validation": "Source dependencies, citation keys, unique labels and figure paths verified",
        "tex_files": len(texfiles), "referenced_figures": images,
        "files": {p.relative_to(SOURCE).as_posix(): sha(p) for p in files},
    }
    (SOURCE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(SOURCE / "manifest.json")
    for name, members, base in [
        ("Report_SingleColumn_LaTeX.zip", files, SOURCE),
        ("Report_Figures.zip", [p for p in files if "figures" in p.relative_to(SOURCE).parts], SOURCE / "figures"),
    ]:
        target = ROOT / "final" / name
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=8) as z:
            for p in members:
                z.write(p, p.relative_to(base).as_posix())
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
        print(f"{target.name}: {len(members)} files; {target.stat().st_size:,} bytes")
    print(f"PASS: {len(images)} figure paths, {len(labels)} labels, {len(cites)} bibliography keys")


if __name__ == "__main__":
    main()
