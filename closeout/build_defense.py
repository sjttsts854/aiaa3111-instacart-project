"""Build the English team rehearsal document from the completed study evidence."""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final"


def build():
    d = Document()
    s = d.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.top_margin = s.bottom_margin = Inches(.65)
    s.left_margin = s.right_margin = Inches(.75)
    for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"]:
        st = d.styles[name]
        st.font.name = "Calibri"
        st.font.color.rgb = RGBColor(0, 0, 0)
    for st in d.styles:
        for border in list(st.element.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    normal = d.styles["Normal"]
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08
    d.styles["Title"].font.size = Pt(26)
    d.styles["Title"].paragraph_format.space_after = Pt(9)
    d.styles["Heading 1"].font.size = Pt(18)
    d.styles["Heading 1"].paragraph_format.space_before = Pt(6)
    d.styles["Heading 1"].paragraph_format.space_after = Pt(8)
    d.styles["Heading 2"].font.size = Pt(12)
    d.styles["Heading 2"].paragraph_format.space_before = Pt(8)
    d.styles["Heading 2"].paragraph_format.space_after = Pt(4)
    header = s.header.paragraphs[0]
    header.text = "AIAA3111  |  Shopping History and Candidate Selection"
    header.runs[0].font.size = Pt(9)
    footer = s.footer.paragraphs[0]
    footer.text = "TEAM REVIEW DRAFT  •  Rehearsal guide  |  "
    footer.runs[0].font.size = Pt(9)
    field = OxmlElement("w:fldSimple"); field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    def p(t, style=None): return d.add_paragraph(t, style)
    def h(t): d.add_heading(t, 2)
    def page(t): d.add_page_break(); d.add_heading(t, 1)
    def qa(q, a): h(q); p(a)

    d.add_heading("Shopping History and Candidate Selection", 0)
    p("English presentation and defense guide", "Subtitle")
    p("Use this guide to explain the finished Instacart study, rehearse the evidence demo, and prepare for questions. The central result is narrow and useful: a smaller candidate set retained average Recall@10 within a predeclared tolerance, while deeper recommendation lists exposed the excluded targets.")
    p("Team number, members and student IDs: pending. Speaking times below are rehearsal suggestions, not a verified course time limit. All team members should review the evidence and adapt the wording before submission or presentation.")
    h("A 90 second opening")
    p("Shopping histories serve two roles in a recommender. They help score products, and they decide which products can enter the recommendation list. We asked whether these roles should use history in the same way.")
    p("We studied next-basket repeat purchases in Instacart. After earlier baseline experiments, we tested a frozen design on 8,000 previously unused users. Every policy was judged against the same full-history repeat target, so removing a candidate could not make the task artificially easier.")
    p("For long-history users, expanding candidates from the last five orders to all past purchases reduced Recall@10 by 3.29 percentage points with cumulative frequency, but increased it by 0.47 points with time-decayed frequency. Old products are therefore useful or harmful depending on how they compete in the ranking.")
    p("A validation-selected policy retained products from the last five orders plus up to five older frequent products. It removed 44.79 percent of candidate volume and retained average Recall@10 within a predeclared half-point tolerance. Follow-up analysis explained why: almost all users had unchanged recall, and excluded true targets usually ranked below position 20. At list lengths 30 and 50, recall losses reached 1.33 and 4.03 points. Our contribution is an auditable explanation of when candidate reduction works and where it stops working.")
    h("The sentence to remember")
    p("The value of shopping history depends on scoring, candidate eligibility and recommendation depth; average retention at Top 10 does not imply that old targets can be discarded without cost.")

    page("A six minute poster walkthrough")
    h("0 to 45 seconds   State the task and puzzle")
    p("We predict which previously purchased products appear in the next basket. The metric is user-average Recall@10, not classification accuracy. Users with at least 16 historical orders initially looked harder than those with 3 to 5: 50.11 versus 67.97 percent recall. But they also had much larger historical catalogs.")
    h("45 to 100 seconds   Explain the comparison problem")
    p("When we compared overlapping history groups using common weights for observed candidate size, the difference reversed. Adding historical mean basket size also produced a positive standardized difference, but retained only 45.82 percent of eligible users. This is evidence that group composition matters. It is not evidence that obtaining more history causally improves prediction.")
    h("100 to 185 seconds   Walk through the controlled experiment")
    p("Point to the two-by-two design. One axis changes the history used for scoring; the other changes candidate eligibility. The same future target stays fixed. Then focus on expansion under fixed full-history scores: frequency loses 3.29 points, whereas decay gains 0.47. The exact hit ledger explains this: frequency finds old true products but displaces even more correct products. With decay, recovered hits exceed displaced hits.")
    h("185 to 255 seconds   Present the selected policy")
    p("Validation chose the last five orders plus at most five older frequent products. On new confirmation users, mean candidate count fell from 64.92 to 35.85. Recall was 58.171 versus 58.164 percent. The simultaneous confidence interval for the difference was minus 0.066 to plus 0.071 percentage points. Its lower bound exceeds the prespecified minus 0.5 margin, supporting noninferiority. The interval crosses zero, so we cannot claim improvement over full candidates.")
    h("255 to 320 seconds   Explain the boundary")
    p("Label this part posthoc. Among 7,472 users with repeat targets, 7,431 had unchanged mean recall; 23 improved and 18 worsened. Only 0.26 percent of excluded true targets originally appeared in the full model's Top 10; 95.54 percent ranked after 20. Longer lists eventually reach these targets: losses grow to 1.33 points at K equals 30 and 4.03 at K equals 50.")
    h("320 to 360 seconds   State the contribution and scope")
    p("We contribute a fixed-target evaluation, a controlled separation of scoring history and candidate competition, and an audited explanation of the tradeoff across list lengths. We do not claim a new state-of-the-art algorithm or measured computational savings. Candidate filtering happened after full-model scores had already been computed.")

    page("Questions about the research contribution")
    qa("What is new in this project", "The contribution is empirical and methodological: we separate two uses of history, measure recovered and displaced hits with fixed targets, confirm a candidate-reduction policy on unused users, and explain its scope with per-user and rank-depth diagnostics. These established ingredients are combined into a coherent investigation; we do not claim to have invented history weighting or candidate pruning.")
    qa("Is this only improving a prediction score", "The main study explains contrasting outcomes under controlled ranking changes. The chosen policy barely changes mean Recall@10. Its useful result is candidate reduction under a declared tolerance, plus evidence showing why that result depends on recommendation depth.")
    qa("Why use Instacart and repeat targets", "The data provide ordered purchases and product identities for many users. Repeat purchases offer a clear candidate universe from observable history. New-item discovery remains outside this task; the full model's recall falls from 58.16 percent for repeat targets to 35.82 percent when all next-basket products form the denominator.")
    qa("Does more history help or hurt", "There is no single answer. Raw groups mix different task sizes. Controlled experiments show that added history and candidates can hurt cumulative frequency but help time-decayed frequency. The effect depends on scoring and eligibility, while observational group contrasts do not identify a causal history effect.")
    qa("Are the old products noise", "No. Full-history candidate expansion helped decayed frequency. Excluded repeat targets also become reachable in longer full-model lists. Low rank under this particular model does not mean a product is unimportant or impossible for another model to recover.")
    qa("Why did the complex methods not become the main story", "Cluster features and rule-based extensions did not show broad, reliable gains beyond strong personal baselines. A small rule-coverage effect did not clearly replicate in a later cohort, and the quota reranker selected zero weight on validation. We retained these results rather than presenting only successful variants.")
    qa("How does the domain baseline fit", "On the separate Study 3 cohort, an independently written TIFU-KNN adaptation gained about 0.61 points over its personal-history counterpart; the full feature model was another 1.96 points higher. The implementation followed documented author-code semantics and our task restrictions, so it is not a reproduction of the paper's benchmark score.")

    page("Questions about statistical and practical validity")
    qa("Did pruning remove difficult users or true labels", "No. Every policy uses the same full-history repeat target for each confirmation user. An excluded true product remains a missed target unless it is still eligible and retrieved. Empty-target users remain in counts and in precision and F1; their recall is undefined.")
    qa("How were leakage and repeated testing handled", "User sets were separated across training, validation and confirmation, and the final cohort excluded users from earlier stages. Global features used training users only; target-order metadata were not predictors. The final policy, models and nine comparisons were frozen before confirmation. Later rank and list-length diagnostics are explicitly posthoc, not independent validation.")
    qa("What exactly does noninferiority mean", "Before confirmation, we defined an acceptable average Recall@10 loss of 0.5 percentage points. The simultaneous lower confidence bound for selected minus full was minus 0.066 points, above that margin. This supports the chosen average-loss criterion on this study population. It does not establish equality, improvement, or a guarantee for every user.")
    qa("Why three model seeds and paired intervals", "Three seeds reduce dependence on one fitted model. We first averaged metrics within each user across the three models; this is not an ensemble of scores. A paired user bootstrap keeps each policy comparison on the same users. The nine primary contrasts used 6,000 resamples and Bonferroni simultaneous 95 percent intervals.")
    qa("Is the frequent old item rule the best policy", "It was noninferior to full candidates and slightly better than a same-budget frequency policy. It did not clearly beat a same-budget recency policy at Top 10. Both policies retain all recent products and have similar candidates and short lists. We therefore claim evidence for selective retention, not that this old-item rule is uniquely optimal.")
    qa("Did anyone become worse off", "Yes. Eighteen of 7,472 users with nonempty repeat targets had lower mean recall under the selected policy, while 23 improved. The remaining 7,431 were unchanged. The harmed users' conditional mean loss was about 21.03 percentage points. Average noninferiority is not individual protection.")
    qa("Did the project reduce runtime or storage", "No measured speedup is claimed. Full-history features and scores were already computed before masking candidates. A future engineering study would move selection before expensive scoring, measure time and memory, and independently validate any performance change.")

    page("Numbers and distinctions to keep straight")
    h("The confirmation population")
    p("Study 4 uses 20,000 training users, 5,000 validation users and 8,000 confirmation users. Recall@10 is averaged over 7,472 confirmation users with nonempty full-history repeat targets. The 528 empty-target users are not assigned zero recall. The long-history factorial comparison uses 2,526 eligible users with at least 16 historical orders. These are different denominators for different questions.")
    h("Percentage points versus percent")
    p("Selected Recall@10 is 58.171 percent; full-candidate recall is 58.164 percent. Their difference is about +0.007 percentage points, not +0.007 percent relative improvement. The 44.79 percent reduction refers to total candidate volume, equivalently the ratio of mean counts 64.92 and 35.85. It is not the mean of individual users' percentage reductions.")
    h("Coverage versus retrieved recall")
    p("Selected candidates retain 90.87 percent of full repeat-target coverage on a user-average basis, while Recall@10 remains about 58.17 percent. Coverage asks whether a target is eligible anywhere in the candidate set; recall asks whether it is retrieved within the first K recommendations. These can change differently when excluded products originally ranked low.")
    h("Confirmation versus posthoc intervals")
    p("The original nine-contrast confirmation interval for selected minus full at K = 10 is [−0.066, +0.071] percentage points. The later list-depth analysis used a separate family of 12 contrasts, so its K = 10 interval is slightly different. Use the original interval for the primary noninferiority claim. Do not describe the follow-up as a new independent confirmation.")
    h("A longer list is a new condition")
    p("Posthoc selected-minus-full losses are 0.308 points at K = 20, 1.334 at K = 30 and 4.028 at K = 50. At K = 20, the simultaneous interval [−0.517, −0.103] slightly crosses the original −0.5 reference margin. This does not invalidate the prespecified Top 10 result, and it does not confirm any newly chosen safe cutoff.")
    h("Within user time versus calendar time")
    p("Each terminal basket follows its own observed history. The dataset does not supply absolute calendar dates, so this design is not a global calendar backtest. Inter-order gaps are capped at 30 days, and prices or purchase amounts are not available.")
    h("Use of artificial intelligence")
    p("Codex assisted research planning, implementation, experiment execution, analysis, visual preparation and drafting. The recorded outputs are from actual local experiments. Team members must verify the code, results and citations, revise the interpretation in their own understanding, and provide their actual contributions. Do not imply that completed student review or contribution assignments have already occurred.")

    page("Evidence demo and team handoff")
    h("A short evidence demonstration")
    p("Extract the portable project archive and follow its README to install the lightweight demo dependencies. From that folder, run python -m streamlit run evidence_app.py. The app reads frozen Study 4 outputs; it does not train a model. Individual examples show seed 42, whereas the report averages per-user metrics over three seeds.")
    for t in [
        "Start with the study overview. Identify the full target, the three policies and the three-seed aggregate results.",
        "Open user 17: full candidates 83, selected candidates 35; both hit 2 of 3 repeat targets. Reveal the six-product next basket to distinguish repeat recall from whole-basket recall.",
        "Show user 4350 improving from 4/8 to 5/8 hits, then user 31433 declining from 1/1 to 0/1. These are the smallest IDs meeting each condition, not representative cases or validated subgroups.",
        "Show user 490 with no repeat targets. Explain why recall is undefined and why the user is still retained in the study. User 1872 also illustrates a changed list with unchanged recall.",
        "Return to the list-depth evidence. Connect the short-list stability to the excluded targets' original ranks and the losses at K = 30 and 50.",
    ]: p(t, "List Bullet")
    h("Checks before the team calls the draft final")
    for t in [
        "Add the actual group number, member names and student IDs to the report and poster; document each member's real contribution.",
        "Check the course platform for any separately issued report template, poster size, submission naming and presentation time. The current report is eight pages and the poster is A1 landscape.",
        "Have every member explain the fixed target, noninferiority margin, same-budget controls and posthoc boundary without reading this guide.",
        "Read every cited source that supports a claim and review the AI-use disclosure. Keep unsupported novelty, speedup and causal claims out of the final wording.",
        "Run the evidence verifier and demo from the extracted package. Keep Report.pdf, source, README and data-download instructions together; submit Poster.pdf separately if the course instructions remain unchanged.",
    ]: p(t, "List Bullet")
    h("Where the evidence lives")
    p("Inside the package, evidence/study4/tables contains the primary comparisons and evidence/study4_followup/tables contains the later user, target-rank and list-depth analyses. README.md and REPRODUCE.md explain inspection and reproduction. Original audits requiring raw data and full caches are distinct from the portable package's integrity and demo checks.")
    OUT.mkdir(exist_ok=True)
    dest = OUT / "Defense_Guide.docx"
    d.save(dest)
    print(dest)


if __name__ == "__main__":
    build()
