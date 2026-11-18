"""Assemble and check the local English project handoff without retraining."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pypdf import PdfReader
from closeout.build_package import finalize_manifest

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final"
PACKAGE = FINAL / "package"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    counts = {}
    for rel in ["autoresearch/research-v4/freeze.json", "autoresearch/research-v4-followup/inputs.json"]:
        manifest = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        for name, expected in manifest["sha256"].items():
            assert sha(ROOT / name) == expected, f"Original frozen file changed: {name}"
        counts[rel] = len(manifest["sha256"])
    pages = {}
    for name, expected in [("Report_Draft.pdf", 8), ("Defense_Guide.pdf", 6), ("Poster_Draft.pdf", 1)]:
        pdf = PdfReader(FINAL / name)
        assert len(pdf.pages) == expected
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        assert not re.search(r"[\u4e00-\u9fff]", text)
        assert len(text) > 1000
        pages[name] = expected
    poster = PdfReader(FINAL / "Poster_Draft.pdf").pages[0]
    assert abs(float(poster.mediabox.width) / 72 * 25.4 - 841) < .1
    assert abs(float(poster.mediabox.height) / 72 * 25.4 - 594) < .1
    shutil.copyfile(FINAL / "Report_Draft.pdf", PACKAGE / "Report.pdf")
    finalize_manifest()
    archive = FINAL / "Group_PENDING_Project_DRAFT.zip"
    selected = [p for p in sorted(PACKAGE.rglob("*")) if p.is_file()
                and not {"__pycache__", ".pytest_cache", ".venv", ".git", "archive"}.intersection(p.parts)]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for path in selected:
            z.write(path, path.relative_to(PACKAGE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert {"Report.pdf", "README.md", "DATA.md", "REPRODUCE.md", "evidence_app.py"} <= set(z.namelist())
        assert "Poster.pdf" not in z.namelist()
        assert z.read("Report.pdf") == (FINAL / "Report_Draft.pdf").read_bytes()
    html = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIAA3111 project handoff</title>
<style>body{max-width:1020px;margin:50px auto;padding:0 25px;color:#172d3d;font:17px/1.6 system-ui,Arial,sans-serif}h1{font-size:38px;line-height:1.2}h2{font-size:23px;margin-top:36px}small{color:#526b78}a{color:#087c75}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:14px;border-bottom:1px solid #d6e0e5;text-align:left}th{background:#edf3f5}code{background:#edf3f5;padding:2px 5px}.badge{font-weight:700;color:#af4b35}li{margin:8px 0}footer{margin-top:40px;padding-top:15px;border-top:1px solid #d6e0e5}</style>
<body><small>AIAA3111 DATA MINING FINAL PROJECT</small>
<h1>Shopping History and Candidate Selection</h1>
<p class="badge">Team review draft</p>
<p>The completed research asks how old purchases should inform product scores and candidate eligibility. A smaller pool retained average Top 10 recall within a prespecified tolerance, while longer lists exposed the lost target coverage.</p>
<h2>Read and edit the deliverables</h2>
<table><thead><tr><th>Deliverable</th><th>Ready to read</th><th>Editable version</th></tr></thead><tbody>
<tr><td>Latest single-column paper · 11 pages</td><td><a href="Report_SingleColumn.pdf">Paper PDF</a></td><td><a href="Report_SingleColumn_LaTeX.zip">LaTeX ZIP for Overleaf</a> · <a href="latex/main.tex">Main source</a></td></tr>
<tr><td>Paper figures · vector PDF and PNG</td><td><a href="Report_Figures.zip">Figure bundle</a></td><td><a href="latex/figures/README.txt">Figure descriptions</a></td></tr>
<tr><td>Earlier Word report · 8 pages</td><td><a href="Report_Draft.pdf">Report PDF</a></td><td><a href="Report_Draft.docx">Word document</a></td></tr>
<tr><td>Academic poster · A1 landscape</td><td><a href="Poster_Draft.pdf">Poster PDF</a></td><td><a href="Poster_Draft.svg">Vector SVG</a></td></tr>
<tr><td>Defense and rehearsal guide · 6 pages</td><td><a href="Defense_Guide.pdf">Guide PDF</a></td><td><a href="Defense_Guide.docx">Word document</a></td></tr>
<tr><td>Portable project · source and evidence</td><td><a href="Group_PENDING_Project_DRAFT.zip">Download ZIP</a></td><td>Extract the ZIP and open its README.md</td></tr>
</tbody></table>
<p>The portable project ZIP contains the earlier eight-page Report.pdf, source code, compact real examples, evidence tables, README and data-download instructions. The latest single-column paper and its editable source are separate downloads above. The poster is separate, as indicated in the provided course guide. Raw data, fitted models and large caches are excluded.</p>
<h2>Run the English evidence demo</h2>
<p>Extract the ZIP, then follow its README for the lightweight demo environment. From the extracted folder, run <code>python -m streamlit run evidence_app.py</code>. It displays frozen outputs for all 8,000 Study 4 confirmation users and performs no model fitting.</p>
<p>Try user 17 for the repeat-versus-whole-basket distinction, 4350 for an improvement, 31433 for harm, and 490 for an empty repeat target. These are illustrative single-model cases. The report averages each user's metrics across three fitted seeds.</p>
<h2>What has been checked</h2>
<ul><li>Eight report pages, six guide pages and the A1 poster were rendered and visually reviewed.</li>
<li>All 108 research tests passed in the packaged source; the five-case demo checks and package metric verifier passed.</li>
<li>The browser demonstration was checked for readable hit columns and persistent view selection.</li>
<li>Original frozen study and follow-up input hashes remain unchanged. The reproduction instructions were source-reviewed; no fresh full-data retraining is claimed.</li></ul>
<h2>Before submission</h2>
<ol><li>Add the actual group number, names, student IDs and individual contributions.</li>
<li>Read, understand and revise the final wording as a team; verify the cited sources and retain an accurate AI-use disclosure.</li>
<li>Check any separately issued report template, poster dimensions, naming rules or presentation timing. The supplied demo guide specifies 6–10 report pages but does not settle all of these details.</li>
<li>Remove draft status only after the team completes its review. Repository sharing does not constitute course submission or peer-reviewed publication.</li></ol>
<footer><small>Primary confirmation and posthoc interpretation remain clearly separated. Candidate count reduction is not a measured runtime saving. See <a href="verification.json">verification record</a> and <a href="SHA256SUMS.txt">file hashes</a>.</small></footer></body></html>'''
    (FINAL / "START_HERE.html").write_text(html, encoding="utf-8")
    for href in re.findall(r'href="([^"]+)"', html):
        if href not in ["verification.json", "SHA256SUMS.txt"]:
            assert (FINAL / href).is_file(), href
    verification = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS for prepared deliverables; team authorship and review pending",
        "pdf_pages": pages, "poster_mm": [841, 594], "new_deliverables_language": "English",
        "original_frozen_files_unchanged": counts,
        "research_tests_passed": 108,
        "portable_demo_checks": "PASS for users 17, 1872, 4350, 31433, 490; three persistent views",
        "browser_qa": "Passed normal case with revealed repeat targets, persistent navigation and visible hit columns",
        "visual_qa": "All report/guide pages and full poster plus column regions inspected",
        "docx_renderer": "Installed Microsoft Word PDF export plus Poppler; bundled render_docx diagnosed missing soffice.exe",
        "clean_full_data_retraining_performed_during_closeout": False,
        "zip": {"name": archive.name, "files": len(selected), "bytes": archive.stat().st_size, "sha256": sha(archive)},
    }
    (FINAL / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    deliverables = [p for p in sorted(FINAL.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (FINAL / "SHA256SUMS.txt").write_text("\n".join(f"{sha(p)}  {p.name}" for p in deliverables) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
