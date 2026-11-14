"""Build the English team-review report from immutable evidence tables.

Run with the Codex bundled Python, then render with the documents skill helper.
This presentation-only builder never trains a model or changes an experiment.
"""
from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'final'; TMP=ROOT/'tmp/closeout_report'; TMP.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True)
S4=ROOT/'reports/study4/tables';FOLLOW=ROOT/'reports/study4_followup/tables'
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def keyed(path,key):return {r[key]:r for r in rows(path)}
def f(row,k):return float(row[k])
primary=keyed(S4/'primary_contrasts.csv','contrast');metrics=keyed(S4/'confirmation_metrics.csv','name')
factor=rows(S4/'factorial_metrics.csv');histor=rows(S4/'history_sensitivity.csv')
depth=rows(FOLLOW/'depth_contrasts.csv')
TEAL='#157b70';BLUE='#405870';RED='#b9553c';GRAY='#6d7580';LIGHT='#e9edef'
FONT=Path('C:/Windows/Fonts')
def font(size,bold=False):return ImageFont.truetype(str(FONT/('arialbd.ttf' if bold else 'arial.ttf')),size)
def text(d,xy,s,size=34,fill='#15202b',bold=False,anchor=None):d.text(xy,s,font=font(size,bold),fill=fill,anchor=anchor)
def chart():
    im=Image.new('RGB',(1800,650),'white');return im,ImageDraw.Draw(im)
def line(d,coords,color=GRAY,width=3):d.line(coords,fill=color,width=width)
def dot(d,x,y,color=TEAL,r=10):d.ellipse([x-r,y-r,x+r,y+r],fill=color)

def charts():
    im,d=chart();xmin,xmax=-20,17;left,right=580,1720
    xx=lambda v:left+(v-xmin)/(xmax-xmin)*(right-left)
    for t in [-20,-10,0,10]:
        line(d,[xx(t),75,xx(t),510],LIGHT);text(d,(xx(t),530),str(t),32,anchor='mt')
    line(d,[xx(0),75,xx(0),510],GRAY)
    raw=float(histor[0]['raw_overall_long_minus_short'])*100
    line(d,[xx(raw),75,xx(raw),510],RED,4)
    for i,r in enumerate([histor[0],histor[2]]):
        y=170+i*220
        text(d,(15,y-15),'Candidate count' if i==0 else 'Count + historical basket size',36,bold=True)
        text(d,(15,y+38),f"Retained: {100*f(r,'retained_fraction'):.1f}% of eligible users",32)
        for dy,col,lo,hi,c in [(-30,'raw_common_support_long_minus_short','raw_common_support_ci95_low','raw_common_support_ci95_high',BLUE),(35,'standardized_long_minus_short','standardized_ci95_low','standardized_ci95_high',TEAL)]:
            v=f(r,col)*100;line(d,[xx(f(r,lo)*100),y+dy,xx(f(r,hi)*100),y+dy],c,7);dot(d,xx(v),y+dy,c)
    text(d,(650,610),'Long minus short history Recall@10 (percentage points)',33,anchor='mm')
    text(d,(30,15),'Raw overall: -17.86 pp',31,RED,bold=True)
    text(d,(600,15),'Same-support raw',31,BLUE,bold=True);text(d,(1060,15),'Same-support standardized',31,TEAL,bold=True)
    im.save(TMP/'fig_history.png')

    im,d=chart()
    for j,method in enumerate(['frequency','decay']):
        x=180+j*890
        text(d,(x+270,20),'Frequency score' if j==0 else 'Time-decayed score',40,bold=True,anchor='mt')
        for iy,sw in enumerate(['5','all']):
            for ix,pool in enumerate(['5','all']):
                r=next(r for r in factor if r['subgroup']=='long_history' and r['method']==method and r['score_window']==sw and r['pool']==pool)
                v=f(r,'recall_at_10');shade=int(230-(v-.43)/.06*100)
                d.rectangle([x+ix*285,125+iy*180,x+(ix+1)*285,125+(iy+1)*180],fill=(shade, min(245,shade+30), min(250,shade+35)),outline='white',width=5)
                text(d,(x+ix*285+142,215+iy*180),f'{100*v:.2f}%',54,bold=True,anchor='mm')
            text(d,(x-20,215+iy*180),'Recent 5' if iy==0 else 'All',31,anchor='rm')
        for ix,label in enumerate(['Recent 5','All']):text(d,(x+ix*285+142,510),label,34,anchor='mt')
        text(d,(x+285,570),'Eligible candidate pool',34,anchor='mt')
    text(d,(15,90),'Scoring history',31,bold=True)
    im.save(TMP/'fig_factorial.png')

    im,d=chart();left,right=620,1700;xmin,xmax=-.55,.40;xx=lambda v:left+(v-xmin)/(xmax-xmin)*(right-left)
    for t in [-.5,-.25,0,.25]:
        line(d,[xx(t),70,xx(t),490],LIGHT);text(d,(xx(t),520),f'{t:g}',32,anchor='mt')
    line(d,[xx(0),70,xx(0),490],GRAY);line(d,[xx(-.5),70,xx(-.5),490],RED,4)
    for i,(key,label) in enumerate([('dual_vs_full','Selected minus full'),('dual_vs_frequency_budget','Selected minus frequency budget'),('dual_vs_recency_budget','Selected minus recency budget')]):
        r=primary[key];y=140+i*145;lo=f(r,'sim_low')*100;hi=f(r,'sim_high')*100;v=f(r,'delta')*100
        text(d,(15,y),label,34,anchor='lm');line(d,[xx(lo),y,xx(hi),y],TEAL,7);dot(d,xx(v),y)
        text(d,(xx(v),y+38),f'{v:+.3f} [{lo:+.3f}, {hi:+.3f}]',28,anchor='mt')
    text(d,(900,610),'Paired Recall@10 difference (percentage points)',34,anchor='mm')
    im.save(TMP/'fig_policy.png')

    im,d=chart();left,right,top,bottom=220,1710,55,500
    xx=lambda k:left+(k-5)/45*(right-left);yy=lambda v:bottom-(v+4.8)/5.4*(bottom-top)
    for t in [0,-1,-2,-3,-4]:
        line(d,[left,yy(t),right,yy(t)],LIGHT);text(d,(left-25,yy(t)),str(t),32,anchor='rm')
    line(d,[left,yy(-.5),right,yy(-.5)],RED,3)
    for control,color in [('full',BLUE),('recency_budget',TEAL)]:
        points=[]
        for r in depth:
            if r['comparator']!=control:continue
            k=int(r['k']);v=f(r,'delta')*100;x=xx(k);y=yy(v);points.append((x,y))
            line(d,[x,yy(f(r,'sim_low')*100),x,yy(f(r,'sim_high')*100)],color,5);dot(d,x,y,color)
        d.line(points,fill=color,width=5)
    for k in [5,10,15,20,30,50]:text(d,(xx(k),530),str(k),33,anchor='mt')
    text(d,(25,25),'Recall difference (pp)',30,bold=True)
    text(d,(680,25),'Selected minus full',31,BLUE,bold=True);text(d,(1190,25),'Selected minus recency',31,TEAL,bold=True)
    text(d,(900,610),'Recommendation list length K',34,anchor='mm');im.save(TMP/'fig_depth.png')

doc=Document();sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65);sec.left_margin=Inches(.75);sec.right_margin=Inches(.75)
sec.header_distance=Inches(.25);sec.footer_distance=Inches(.3)
for name,size in [('Normal',11),('Title',22),('Subtitle',11),('Heading 1',14),('Heading 2',11.5),('Caption',9.2)]:
    st=doc.styles[name];st.font.name='Calibri';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
    st.paragraph_format.space_after=Pt(6);st.paragraph_format.line_spacing=1.08
doc.styles['Normal'].paragraph_format.widow_control=True
for name in ['Heading 1','Heading 2']:
    doc.styles[name].paragraph_format.space_before=Pt(9);doc.styles[name].paragraph_format.keep_with_next=True
doc.styles['Title'].paragraph_format.space_after=Pt(7)
for border in doc.styles.element.xpath('.//w:pBdr'):
    border.getparent().remove(border)
header=sec.header.paragraphs[0];header.text='AIAA 3111  |  Shopping history and candidate competition';header.style=doc.styles['Caption']
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('Team review draft   |   ')
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
for r in footer.runs:r.font.size=Pt(9)
def p(s,style=None):return doc.add_paragraph(s,style=style)
next_page=False
def h(s):
    global next_page
    q=doc.add_heading(s,level=1)
    if next_page:q.paragraph_format.page_break_before=True;next_page=False
    return q
def sub(s):return doc.add_heading(s,level=2)
def page():
    global next_page
    next_page=True
def figure(name,caption):
    q=doc.add_paragraph();q.paragraph_format.keep_with_next=True;q.add_run().add_picture(str(TMP/name),width=Inches(7))
    cap=p(caption,'Caption');cap.paragraph_format.space_after=Pt(8)
def table(headers,data,widths):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    pr=t._tbl.tblPr;borders=OxmlElement('w:tblBorders')
    for n in ['top','left','bottom','right','insideH','insideV']:
        x=OxmlElement('w:'+n);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');borders.append(x)
    pr.append(borders)
    for j,s in enumerate(headers):t.rows[0].cells[j].text=s
    for row in data:
        for c,s in zip(t.add_row().cells,row):c.text=str(s)
    for i,row in enumerate(t.rows):
        for j,c in enumerate(row.cells):
            c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc=c._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'34495E' if i==0 else ('F0F3F5' if i%2==0 else 'FFFFFF'));tc.append(sh)
            mar=OxmlElement('w:tcMar')
            for side in ['top','bottom','left','right']:
                v=OxmlElement('w:'+side);v.set(qn('w:w'),'85');v.set(qn('w:type'),'dxa');mar.append(v)
            tc.append(mar)
            for q in c.paragraphs:
                q.paragraph_format.space_after=Pt(1);q.paragraph_format.line_spacing=1.0
                q.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in q.runs:r.font.size=Pt(9.5);r.font.bold=i==0;r.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '000000')
        trp=row._tr.get_or_add_trPr();n=OxmlElement('w:cantSplit');trp.append(n)
    repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
    p('').paragraph_format.space_after=Pt(1)
    return t
def formula():
    q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER
    m=OxmlElement('m:oMathPara');math=OxmlElement('m:oMath')
    def mr(parent,s):
        r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=s;r.append(t);parent.append(r)
    mr(math,'Recall@K(u) = ');frac=OxmlElement('m:f');num=OxmlElement('m:num');den=OxmlElement('m:den')
    mr(num,'|TopK(u) ∩ R(u)|');mr(den,'|R(u)|');frac.append(num);frac.append(den);math.append(frac);m.append(math);q._p.append(m)

def build():
    charts()
    p('Shopping History and Candidate Competition in Next Basket Recommendation','Title')
    p('AIAA 3111 course project  |  English team review draft','Subtitle')
    p('Team number, member names and student IDs: pending team completion.','Caption')
    sub('Abstract')
    p('Purchase histories provide both ranking signals and the catalog of products eligible for recommendation. We separate these roles in an Instacart repeat-purchase task while keeping users and true targets fixed. A new cohort contains 20,000 training, 5,000 validation and 8,000 confirmation users. Among 2,526 long-history users with nonempty targets, expanding old candidates reduces Recall@10 by 3.29 percentage points under cumulative frequency but improves it by 0.47 points under time decay. A validation-selected pool of recent products plus five older frequent products reduces total candidates by 44.79%. Its confirmation Recall@10 is 58.171%, compared with 58.164% for all historical candidates, satisfying a prespecified 0.5-point noninferiority margin. Subsequent, explicitly posthoc analysis finds unchanged recall for 99.45% of eligible users. Excluded true targets mostly occupy low original ranks, and losses grow for longer lists. The contribution is a controlled empirical analysis of historical information, candidate competition and the limits of selective retention, rather than a claim of a new best-performing algorithm.')
    h('1 Introduction')
    p('A grocery recommender must choose a short list from a user’s past purchases. Keeping more history may reveal recurring preferences, but it also permits more old products to compete for limited positions. Aggregate accuracy alone cannot distinguish these effects. In particular, users with more observed orders also have larger historical catalogs, making comparisons across history-length groups difficult to interpret.')
    p('Earlier experiments in this project established strong personal-history baselines. Time weighting helped, while adding clustering and association-rule features did not reliably improve the full model. This motivated three questions: how sensitive are history-group comparisons to their composition; what happens when scoring history and candidate eligibility change separately; and how much candidate reduction can a fixed short-list ranker tolerate?')
    sub('Relation to prior work and contribution')
    p('TIFU-KNN models personalized item frequencies and neighboring users [1]. A next-basket reality check distinguishes repeat and exploratory recommendations and emphasizes careful evaluation [2]. We use these ideas as context and compare an adapted TIFU baseline in an earlier cohort. Our contribution is a reproducible course-level study of the two roles of history, with fixed-target evaluation, exact hit accounting and candid reporting of null results and scope limits.')

    page();h('2 Task and evidence design')
    sub('2 1 Data and prediction boundary')
    p('The public Instacart data contain 206,209 users, 3,421,083 orders and 49,688 products. Six CSV files provide users’ order sequences, basket contents and product metadata [4,5]. We use the 131,209 users whose terminal order has released product labels. Earlier orders form the observed history. The prediction point is after the last historical order, before any metadata from the terminal order is available.')
    p('For user u, C_all(u) is the set of every product purchased in the observed history. B_next(u) is the terminal basket, and R(u) = B_next(u) ∩ C_all(u) is the fixed repeat-target set. A policy chooses an eligible subset of C_all(u), then returns at most K products. Excluded targets remain misses; pruning never changes R(u) or the user population.')
    formula()
    p('Recall is averaged across users with nonempty R(u). The main confirmation sample has 8,000 users, including 528 empty repeat-target users; hence 7,472 contribute to macro Recall and NDCG. Precision and F1 retain all users. We also report hits divided by K, actual recommendation length, coverage and the top-K oracle bound, so shorter lists cannot create a misleading precision improvement.')
    sub('2 2 Separate exploration from confirmation')
    table(['Stage','Train','Validation','Held out','Role'],[
        ['Initial study','7,000','1,500','1,500','Pipeline and hypotheses'],
        ['Information study','20,000','5,000','8,000','Feature comparisons'],
        ['Baseline extension','20,000','5,000','8,000','TIFU and robustness'],
        ['Memory study','20,000','5,000','8,000','Main confirmation'],
        ['Posthoc follow-up','No new fitting','No selection','Same last 8,000','Interpretation and scope']], [1.65,.75,.9,.85,1.85])
    p('The four cohorts are mutually disjoint. The main memory study excludes all 76,000 previously used users before sampling a fresh 33,000. Different-cohort scores are not an improvement trajectory. The follow-up reuses the completed confirmation predictions and is never counted as another independent test.')
    sub('2 3 Leakage controls and data limits')
    p('Global statistics, clustering, rules and neighbor banks use training users’ historical data only. Held-out users supply their own past purchases, not other held-out users as neighbors. No future order day, hour or interval enters a feature. Candidate sets include all observed historical products without negative sampling. Absolute calendar dates and prices are absent, and inter-order gaps are capped at 30 days. This is neither calendar-based backtesting nor a spending analysis.')

    page();h('3 Methods and evaluation')
    sub('3 1 Separate ranking information from eligibility')
    p('The core 2 by 2 comparison crosses a scoring window (recent five orders or all observed orders) with an eligible pool (products in the recent five orders or all historical products). Every cell still knows the full historical catalog and uses the same terminal targets. Thus the experiment changes algorithmic use of history, not how much history the user actually possesses.')
    p('For frequency scoring, the score is the purchase count within the specified window. For time-decayed scoring, each purchase contributes 0.8 raised to its age in orders; age zero is the latest observed order. Counts outside a recent scoring window contribute zero. The primary factorial population has at least 16 historical orders and nonempty repeat targets: 2,526 users in every comparison. Windows of three and ten orders are sensitivity checks.')
    p('Two principal contrasts answer distinct questions. The information contrast adds old purchases to scoring while keeping the full candidate pool. The expansion contrast enlarges the pool while keeping all-history scores fixed. We also report their interaction. Recent-window scoring assigns all old products zero, so some interaction structure is built into the design; it is not an unexpected behavioral discovery.')
    sub('3 2 Learned ranking and candidate policies')
    p('A LightGBM classifier [3] scores user-product pairs using frequency, recency, basket history, purchase-gap, product and department-affinity features. Fixed settings are 240 trees, learning rate 0.05, 31 leaves, minimum child size 80, feature fraction 0.9 and L2 penalty 1. Models use seeds 42, 73 and 123. Policies share each fitted model’s scores; the reported values average per-user metrics across seeds rather than ensemble prediction scores.')
    p('The selective pool retains every product in the recent five orders and adds up to M old products, ordered by all-history count, then recency, then product ID. Validation tests M = 0, 5, 10, 20, 40 and 80. We choose the smallest mean pool whose paired marginal 95% Recall@10 lower bound against the full pool is at least −0.005; otherwise we retain the full pool. Two controls use exactly the same candidate count for each user, selecting by frequency or by recency. Ties remain deterministic.')
    sub('3 3 Uncertainty and exact hit accounting')
    p('The main study prespecifies nine contrasts and uses 6,000 paired-user bootstrap replicates, with marginal and Bonferroni simultaneous 95% intervals. Final noninferiority requires the simultaneous lower bound to exceed or equal −0.005. The validation search is exploratory; the independent confirmation determines the final claim. Intervals condition on fitted models and do not include retraining uncertainty.')
    p('Under fixed scores, enlarging a pool can add correct recommendations and displace correct recommendations from the smaller pool. For each user, new true hits minus displaced old true hits exactly equals the total hit change. Dividing each term by the same R(u) before averaging preserves this identity for macro recall. This is an algorithmic accounting identity, not a causal model of shopper psychology.')

    page();h('4 Baselines and the history length puzzle')
    sub('4 1 Earlier comparisons establish the context')
    table(['Within-cohort comparison','Recall@10 difference','Interpretation'],[
        ['Time decay minus frequency','+3.73 pp','Time information helps'],
        ['Full minus no temporal features','+3.89 pp','Conditional temporal value'],
        ['TIFU minus same-time personal vector','+0.61 pp','Small neighbor contribution'],
        ['Full minus adapted TIFU','+1.96 pp','Strong learned baseline']], [3.0,1.15,2.85])
    p('The first two comparisons use the information cohort; the latter two use the baseline-extension cohort. Clustering and rules had no stable overall benefit, and an early local rule signal did not reproduce clearly. A quota-constrained reranker selected weight zero on validation, leaving the ranking unchanged. Its equal confirmation score is a structural identity, not evidence of success.')
    p('The TIFU adaptation follows released author-code time weighting and historical candidates, not the paper’s full-basket benchmark. Its neighbor contrast uses the same selected personal time vector. Rules and neighbors start from different personal baselines, preventing attribution of their differences solely to representation.')
    sub('4 2 Historical group composition changes the comparison')
    p('In the main memory study, short-history users (3–5 orders) have mean Recall@10 of 67.97%, versus 50.11% for long-history users (16+ orders). Their mean candidate counts are 27.66 and 111.48. We restrict to common historical-covariate cells, then apply identical pooled cell weights to each history group. Figure 1 separates the change in population from the subsequent reweighting.')
    figure('fig_history.png','Figure 1. Long minus short history difference. Historical count-only and count-plus-historical-basket-size strata retain 79.9% and 45.8% of nonempty-target users. Bars are conditional marginal 95% bootstrap intervals. Minimum cell counts of 10 and 20 give the same support here, so four planned settings yield two distinct results.')
    p('The raw overall gap is −17.86 points. Same-support raw gaps remain negative (−5.68 and −8.06); standardized gaps become +13.15 and +4.21. This demonstrates sensitivity to comparison composition, not a causal benefit of more history. No future target size defines these strata, although Recall still conditions inclusion on nonempty future repeat targets. Coarse bins and support restriction limit generalization.')

    page();h('5 Old history helps or hurts depending on scoring')
    figure('fig_factorial.png','Figure 2. Macro Recall@10 for the same 2,526 long-history users. Rows change scoring history; columns change candidate eligibility. Every cell uses the complete repeat-target denominator and knows the full historical catalog.')
    p('All four principal factorial contrasts have main-study simultaneous intervals excluding zero. Both the effect of old scoring information and the effect of eligible old candidates depend on the scoring rule, with opposite directions under frequency and decay.')
    table(['Main factorial contrast','Difference in pp','Simultaneous 95% interval'],[
        ['Frequency scoring information','−2.51','[−3.85, −1.16]'],
        ['Decay scoring information','+2.03','[+1.29, +2.73]'],
        ['Frequency candidate expansion','−3.29','[−4.36, −2.26]'],
        ['Decay candidate expansion','+0.47','[+0.14, +0.83]']], [3.25,1.35,2.4])
    sub('5 1 What changes inside the top ten')
    table(['Fixed-score expansion','New true hits contribution','Displaced true hits contribution','Net recall change'],[
        ['Frequency recent to full','+3.66 pp','−6.94 pp','−3.29 pp'],
        ['Decay recent to full','+0.61 pp','−0.14 pp','+0.47 pp'],
        ['Learned recent to full','+1.45 pp','−0.58 pp','+0.87 pp']], [2.5,1.5,1.7,1.3])
    p('The ledger uses the same long-history population and full repeat-target denominator. Rounding can affect the displayed sum; stored per-user values balance exactly. Frequency retrieves some old true targets but loses more recall from displaced recent targets. Decay admits fewer old products to the front of the list and obtains a positive net balance. These observations reject a blanket recommendation to remove all old history.')
    p('The frequency interaction is negative; the decay interaction interval includes zero. Because recent scoring zeros old items, some interaction is structural. The learned-model ledger is descriptive, not an additional prespecified test.')

    page();h('6 Selective retention preserves short list performance')
    p('Validation selects M = 5. Recent-only candidates fail the selection criterion; adding up to five old frequent products is the smallest tested qualifying pool. The selection, model scores and evaluation code are frozen before confirmation. The confirmation comparison therefore tests a fixed policy rather than choosing the best curve after viewing its outcomes.')
    table(['Candidate policy','Mean candidates','Recall@10','Target coverage'],[
        ['Full historical pool','64.92','58.164%','100.00%'],
        ['Recent five orders only','32.49','57.674%','87.06%'],
        ['Recent plus five old frequent','35.85','58.171%','90.87%'],
        ['Same-budget frequency','35.85','57.980%','89.91%'],
        ['Same-budget recency','35.85','58.110%','90.48%']], [3.1,1.2,1.2,1.5])
    figure('fig_policy.png','Figure 3. Main-study paired contrasts on 7,472 nonempty-target users. Bars are simultaneous 95% intervals across the nine prespecified comparisons. The red −0.5 pp margin applies to selected versus full noninferiority; it is not a superiority threshold for every contrast.')
    p('Total eligible candidates decrease by 44.79%, equivalent to the ratio of mean counts for the same 8,000 users. The mean of individual retention ratios is 73.01%, a different aggregation. Selected minus full recall is +0.0072 points, with simultaneous interval [−0.0657, +0.0713]. This supports noninferiority at the stated margin, but not superiority or exact equality. The improvement over frequency-budget selection is +0.192 points; the +0.061-point difference from recency-budget selection is uncertain.')
    sub('6 1 Retaining recall does not retain all demand')
    p('Macro target coverage falls to 90.87%, and the candidate-constrained oracle Recall@10 falls from 93.34% to 86.09%. Excluded true targets remain in evaluation. The current ranker does not recover most of them in its original top ten, so coverage loss and almost unchanged achieved recall can coexist. This does not make those products unimportant.')
    p('The full model’s whole-basket Recall@10 is only 35.82%, compared with 58.16% for repeat targets. The experiment does not solve first-time product recommendation. It also computes full scores before masking candidates, so candidate reduction is not a measurement of faster scoring, lower memory use or end-to-end deployment savings.')

    page();h('7 Posthoc interpretation and list length limits')
    p('After the main confirmation completed, we inspected unchanged frozen predictions without fitting models or selecting new parameters. All results on this page are posthoc. Among 7,472 nonempty-target users, 7,431 (99.45%) have unchanged seed-averaged recall, 23 improve and 18 decline. Across all 8,000 users, 96.225% retain exactly the same top-ten list under all three models. Thus preservation is mostly explained by unchanged recommendations, not broad improvements cancelling broad losses.')
    p('Some individuals still lose: the 18 harmed users have a conditional mean recall decline of 21.03 points. This outcome-selected group cannot be identified as a validated prediction-time target population. Historical subgroup intervals provide no stable evidence for a beneficial deployment subgroup.')
    figure('fig_depth.png','Figure 4. Fixed-policy differences at six inspected list lengths. Intervals use 6,000 paired-user bootstrap samples with simultaneous adjustment over 12 posthoc contrasts. The red line is the original −0.5 pp reference, not a newly confirmed safe-K boundary. Lines connect tested values only.')
    table(['List length K','Selected minus full','Posthoc simultaneous 95% interval'],[
        ['10','+0.007 pp','[−0.078, +0.070]'],
        ['20','−0.308 pp','[−0.517, −0.103]'],
        ['30','−1.334 pp','[−1.645, −1.023]'],
        ['50','−4.028 pp','[−4.471, −3.606]']], [1.6,2.3,3.1])
    p('Of 5,274 excluded true repeat targets, only 0.26% originally rank in the full model’s top ten, while 95.54% rank after position 20, averaged across seeds. Longer full-pool lists begin to reach these targets, so deletion losses exceed gains from promoted retained products. K=20 slightly crosses the original margin in this posthoc interval; this does not reverse the original K=10 confirmation.')
    p('The simple recency-budget control remains close because both policies preserve all recent products. Overall candidate Jaccard averages 92.92%, but old-only Jaccard is 53.35% among users with a nonempty old union. For 94.61% of all users, all three models give identical top tens. At K=30/50, selected retention modestly exceeds recency-budget recall, yet still loses to the full pool. Such local benefits do not establish overall superiority.')

    page();h('8 Conclusions and limitations')
    p('History has two separable algorithmic roles: it informs scores and supplies eligible competitors. Controlled comparisons show that old candidates can either help or hurt, depending on time weighting. A frozen recent-plus-old-frequent pool preserves top-ten recall within a specified margin while reducing candidate count. Posthoc evidence explains why: most short lists remain unchanged, while omitted true targets concentrate at low original ranks. The cost becomes clearer when the list grows.')
    p('The contribution is a course-level empirical design, a transparent hit decomposition and a measured performance-versus-coverage tradeoff. It is not a new state-of-the-art algorithm, a causal account of human forgetting, or proof that all older products should be removed. Negative rule results, the inactive quota proposal and the strong recency-budget control remain part of the evidence.')
    p('Limitations include one historical platform, terminal-order evaluation, missing prices and absolute dates, historical-product-only candidates and no online user study. Three model seeds do not replace resampling training data. Standardization depends on common support and coarse bins; intervals condition on fitted models. List-length and subgroup follow-ups are posthoc. No measured speedup, sales effect or satisfaction improvement is claimed. A stronger ranker could make presently low-ranked excluded targets more valuable.')
    sub('Reproducibility and team review')
    p('The repository preserves protocols, disjoint splits, source hashes, validation choices, frozen scores, tables and independent audits. The main study checks raw baskets and fixed-target evaluation; the follow-up independently reproduces rankings, ledgers and intervals. The combined test suite passes 108 tests. These checks support implementation consistency, not external generalization. The packaged README distinguishes inspecting saved evidence from rerunning training.')
    sub('References')
    refs=[
    '[1] H. Hu, X. He, J. Gao and Z.-L. Zhang. Modeling Personalized Item Frequency Information for Next-basket Recommendation. SIGIR, 2020, pp. 1071–1080. https://arxiv.org/abs/2006.00556. Author implementation: https://github.com/HaojiHu/TIFUKNN.',
    '[2] M. Li, S. Jullien, M. Ariannezhad and M. de Rijke. A Next Basket Recommendation Reality Check. arXiv:2109.14233, version 2, 2023 (initial version 2021). https://arxiv.org/abs/2109.14233v2.',
    '[3] G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017. proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html.',
    '[4] Instacart. Online Grocery Shopping Dataset 2017. Data dictionary by Jeremy Stanley: https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b.',
    '[5] Yasser H. Instacart Online Grocery Basket Analysis Dataset. Kaggle distribution used locally: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset. CSV fingerprints are recorded; a mirror revision number is not inferred from file contents.'
    ]
    for ref in refs:
        q=p(ref);q.paragraph_format.space_after=Pt(3)
        for r in q.runs:r.font.size=Pt(9)
    sub('AIGC disclosure and pending authorship')
    q=p('OpenAI Codex assisted research design, implementation, experiment execution, analysis, visualization and drafting. Automated tests and independent computational checks were run; team review is still required. This is a review draft, not a claim that students independently authored or verified all content. Before submission, the team must verify the work, revise its own final expression, supply member identities and actual contributions, and disclose assistance according to the course policy.')
    for r in q.runs:r.font.size=Pt(9.5)
    doc.core_properties.title='Shopping History and Candidate Competition in Next Basket Recommendation'
    doc.core_properties.subject='AIAA 3111 team review draft'
    doc.core_properties.author='Team details pending'
    doc.save(OUT/'Report_Draft.docx')
    for border in doc.element.xpath('.//w:pBdr'):
        border.getparent().remove(border)
    doc.save(OUT/'Report_Draft.docx')
    content=[q.text for q in doc.paragraphs]
    (ROOT/'closeout/report_content.txt').write_text('\n'.join(content),encoding='utf-8')
    print(OUT/'Report_Draft.docx')

if __name__=='__main__':build()
