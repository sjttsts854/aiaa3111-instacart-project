"""Build an editable vector poster from audited local evidence tables."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle, PolyLine
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, Color, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'final'; QA=ROOT/'tmp/closeout_poster'
W=841*72/25.4;H=594*72/25.4
M=66;GAP=42;CW=(W-2*M-2*GAP)/3
NAVY=HexColor('#142d42');TEAL=HexColor('#117d75');RUST=HexColor('#b6543e')
INK=HexColor('#172c3b');GRAY=HexColor('#4b5d68');LIGHT=HexColor('#eef4f5');LINE=HexColor('#cfdbdf')

def rows(path):
    with (ROOT/path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def build():
    OUT.mkdir(exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold','C:/Windows/Fonts/arialbd.ttf'))
    d=Drawing(W,H);bounds=[]
    def rect(x,t,w,h,fill,stroke=None):
        d.add(Rect(x,H-t-h,w,h,fillColor=fill,strokeColor=stroke,strokeWidth=1))
    def line(x1,t1,x2,t2,color=LINE,width=1):
        d.add(Line(x1,H-t1,x2,H-t2,strokeColor=color,strokeWidth=width))
    def text(s,x,t,size=26,color=INK,bold=False,anchor='start',width=None):
        font='Arial-Bold' if bold else 'Arial';sw=pdfmetrics.stringWidth(str(s),font,size)
        if width is not None and sw>width+.1:raise ValueError(f'Overwide text {s}: {sw}>{width}')
        x0=x if anchor=='start' else x-sw if anchor=='end' else x-sw/2
        if x0<-.1 or x0+sw>W+.1 or t<0 or t+size>H:raise ValueError(f'Page overflow: {s}')
        d.add(String(x,H-t-size,str(s),fontName=font,fontSize=size,fillColor=color,textAnchor=anchor))
        bounds.append({'text':str(s),'x':x0,'top':t,'width':sw,'height':size})
    def para(s,x,t,w=CW,size=25,leading=33,color=INK,bold=False):
        font='Arial-Bold' if bold else 'Arial';words=s.split();acc='';lines=[]
        for word in words:
            nxt=(acc+' '+word).strip()
            if pdfmetrics.stringWidth(nxt,font,size)>w and acc:lines.append(acc);acc=word
            else:acc=nxt
        if acc:lines.append(acc)
        for j,v in enumerate(lines):text(v,x,t+j*leading,size,color,bold,width=w)
        return t+len(lines)*leading
    def tag(s,x,t,fill):
        tw=pdfmetrics.stringWidth(s,'Arial-Bold',18)+24
        rect(x,t,tw,31,fill);text(s,x+12,t+4,18,white,True)
    def section(n,title,x,t):
        text(n,x,t,39,TEAL,True);text(title,x+49,t+4,31,NAVY,True,width=CW-49)
        line(x,t+57,x+CW,t+57,TEAL,2)
    def bullet(s,x,t,w=CW,size=24):
        d.add(Circle(x+4,H-t-13,3.8,fillColor=TEAL,strokeColor=None))
        return para(s,x+18,t,w-18,size,31)

    metrics={r['name']:r for r in rows('reports/study4/tables/confirmation_metrics.csv')}
    protocol=json.loads((ROOT/'reports/study4/protocol.json').read_text(encoding='utf-8'))
    primary={r['contrast']:r for r in rows('reports/study4/tables/primary_contrasts.csv')}
    factorial={r['name']:r for r in rows('reports/study4/tables/factorial_metrics.csv') if r['subgroup']=='long_history'}
    depth=[r for r in rows('reports/study4_followup/tables/depth_contrasts.csv') if r['comparator']=='full']
    change=next(r for r in rows('reports/study4_followup/tables/users_change_summary.csv') if r['comparator']=='full')
    fullusers=int(change['users']);nrecall=int(change['nonempty_users']);nempty=int(change['empty_target_users'])
    fact=next(iter(factorial.values()));nlong=int(fact['users'])-int(fact['empty_repeat_targets'])
    targetrows=[r for r in rows('reports/study4_followup/tables/depth_excluded_target_ranks.csv') if r['eligibility']=='excluded']
    target_total=sum(float(r['targets']) for r in targetrows)
    tail20=sum(float(r['targets']) for r in targetrows if r['rank_band'] in ['21-50','>50'])/target_total
    top10=sum(float(r['targets']) for r in targetrows if r['rank_band']=='1-10')/target_total
    reduction=1-float(metrics['dual']['mean_candidates'])/float(metrics['full']['mean_candidates'])
    # Header and fixed-evaluation strip.
    rect(0,0,W,244,NAVY)
    text('AIAA 3111 COURSE PROJECT',M,29,21,HexColor('#b8dad8'),True)
    text('TEAM REVIEW DRAFT',W-M,30,20,white,True,'end')
    text('What should a grocery recommender remember?',M,72,65,white,True,width=W-2*M)
    text('Scoring history and candidate competition',M,158,34,white)
    text('Group, member names and student IDs: to be completed',M,208,21,HexColor('#d7e7ea'))
    rect(M,270,W-2*M,117,LIGHT)
    text('Predict the next basket using previously purchased products [1]',M+23,282,28,NAVY,True)
    text(f"Disjoint users: {protocol['train_users']:,} training, {protocol['validation_users']:,} validation, {fullusers:,} confirmation; none used in earlier experiments.",M+23,324,24,INK)
    text(f'Recall@10 = hits / all repeat targets, averaged over {nrecall:,} users; {nempty:,} have no repeat targets and undefined recall. Targets and users stay fixed.',M+23,356,24,GRAY,width=W-2*M-46)

    x1=M;x2=M+CW+GAP;x3=M+2*(CW+GAP)
    line(x2-GAP/2,417,x2-GAP/2,1460,LINE,1)
    line(x3-GAP/2,417,x3-GAP/2,1460,LINE,1)
    # Column 1: controlled factorial experiment.
    section('1','Separate the roles of history',x1,418)
    tag('INDEPENDENT CONFIRMATION',x1,493,TEAL)
    para('Does older history help scoring, or change which products compete for the next ten places?',x1,539,size=26,leading=35)
    para(f'Each cell: Recall@10 for the same {nlong:,} users with at least 16 prior orders and nonempty targets.',x1,655,size=23,leading=30,color=GRAY)

    def matrix(method,title,top):
        text(title,x1,top,28,NAVY,True)
        gx=x1+224;gy=top+80;cellw=(CW-224)/2;cellh=83
        text('Eligible candidates',gx+cellw,top+39,21,GRAY,False,'middle')
        text('Recent 5',gx+cellw/2,top+68,21,INK,True,'middle')
        text('All history',gx+cellw*1.5,top+68,21,INK,True,'middle')
        gy+=21
        for i,score in enumerate(['5','all']):
            text('Recent 5 scores' if score=='5' else 'All-history scores',x1,gy+i*cellh+27,22,INK)
            for j,pool in enumerate(['5','all']):
                value=float(factorial[f'{method}_s{score}_c{pool}']['recall_at_10'])*100
                color=HexColor('#e5f0ef') if i==0 else HexColor('#c7dfdb')
                if i==1 and j==1:color=HexColor('#f2d9d1') if method=='frequency' else HexColor('#a7cec6')
                rect(gx+j*cellw,gy+i*cellh,cellw,cellh,color,white)
                text(f'{value:.2f}%',gx+(j+.5)*cellw,gy+i*cellh+21,34,NAVY,True,'middle')
        return gy+2*cellh
    end1=matrix('frequency','Purchase frequency',738)
    end2=matrix('decay','Order-decayed frequency (0.8)',1042)
    # Numeric contrast lies beneath the matrices and is explicitly conditional.
    text('Expand candidates with all-history scores fixed',x1,1340,22,NAVY,True,width=CW)
    v1=float(primary['frequency_expansion']['delta'])*100;v2=float(primary['decay_expansion']['delta'])*100
    text(f'{v1:+.2f} pp',x1,1378,42,RUST,True)
    text(f'{v2:+.2f} pp',x1+CW*.55,1378,42,TEAL,True)
    text('Frequency',x1,1430,22,GRAY);text('Order-decayed',x1+CW*.55,1430,22,GRAY)

    # Column 2: practical frozen candidate policy.
    section('2','Preserve a short recommendation list',x2,418)
    tag('INDEPENDENT CONFIRMATION',x2,493,TEAL)
    para('Selected on validation: recent-five-order products plus up to five older products with the highest historical frequency.',x2,539,size=25,leading=34)
    text('Fixed LightGBM scores [2]; three fitted seeds.',x2,619,23,GRAY,width=CW)
    text(f'{100*reduction:.2f}%',x2,660,75,TEAL,True)
    text('fewer candidates in total',x2,744,28,NAVY,True)
    text(f"Mean per user: {float(metrics['full']['mean_candidates']):.2f} to {float(metrics['dual']['mean_candidates']):.2f}",x2,785,25,GRAY)
    text(f'Candidate means: {fullusers:,} users; recall: {nrecall:,} users.',x2,818,20,GRAY,width=CW)
    table_y=847
    rect(x2,table_y,CW,44,NAVY)
    text('Candidate policy',x2+12,table_y+7,21,white,True)
    text('Mean size',x2+CW*.63,table_y+7,21,white,True,'middle')
    text('Recall@10',x2+CW-12,table_y+7,21,white,True,'end')
    labels=[('full','Full history'),('recent','Recent only'),('dual','Recent + old frequent'),('frequency_budget','Matched frequency'),('recency_budget','Matched recency')]
    for i,(key,label) in enumerate(labels):
        top=table_y+44+i*48
        rect(x2,top,CW,48,HexColor('#e3f0ed') if key=='dual' else (white if i%2 else LIGHT))
        text(label,x2+12,top+10,21,INK,key=='dual')
        text(f"{float(metrics[key]['mean_candidates']):.2f}",x2+CW*.63,top+10,22,INK,key=='dual','middle')
        text(f"{100*float(metrics[key]['recall_at_10']):.3f}%",x2+CW-12,top+10,22,INK,key=='dual','end')
    contrast=primary['dual_vs_full']
    text('Performance retention confirmed',x2,1163,28,NAVY,True)
    para(f"Selected minus full: {float(contrast['delta'])*100:+.3f} pp. Simultaneous 95% CI [{float(contrast['sim_low'])*100:+.3f}, {float(contrast['sim_high'])*100:+.3f}] pp.",x2,1208,size=25,leading=34)
    para(f"The lower bound exceeds the prespecified -{100*protocol['practical_margin']:g} pp noninferiority margin.",x2,1285,size=25,leading=34)
    para(f"No confirmed advantage over full or matched-recency candidates. Target coverage falls to {100*float(metrics['dual']['candidate_coverage']):.2f}%.",x2,1365,size=25,leading=34,color=RUST)

    # Column 3: posthoc interpretation and list length.
    section('3','Find where pruning loses targets',x3,418)
    tag('POSTHOC ANALYSIS',x3,493,RUST)
    text(f"{100*float(change['unchanged_fraction']):.2f}%",x3,540,66,TEAL,True)
    text(f"of {int(change['nonempty_users']):,} users keep the same Recall@10",x3,617,25,NAVY,True,width=CW)
    text(f"{int(change['improved_users']):,} improve, {int(change['harmed_users']):,} decline, {int(change['unchanged_users']):,} are unchanged.",x3,657,24,GRAY,width=CW)
    text('Longer lists expose the coverage loss',x3,716,28,NAVY,True,width=CW)
    # K-loss curve with actual discrete simultaneous intervals.
    cx=x3+83;ct=779;pw=CW-113;ph=360
    ymin=-5;ymax=.6
    px=lambda k:cx+(k-5)/45*pw
    py=lambda value:ct+(ymax-value)/(ymax-ymin)*ph
    for tick in [0,-1,-2,-3,-4,-5]:
        yy=py(tick);line(cx,yy,cx+pw,yy,LINE,1);text(f'{tick:+d}' if tick else '0',cx-16,yy-12,21,GRAY,False,'end')
    line(cx,ct,cx,ct+ph,GRAY,1.3);line(cx,ct+ph,cx+pw,ct+ph,GRAY,1.3)
    pts=[]
    for r in depth:
        k=int(r['k']);value=float(r['delta'])*100;lo=float(r['sim_low'])*100;hi=float(r['sim_high'])*100
        xx=px(k);yy=py(value);pts.extend([xx,H-yy])
        line(xx,py(lo),xx,py(hi),RUST,2);line(xx-6,py(lo),xx+6,py(lo),RUST,2);line(xx-6,py(hi),xx+6,py(hi),RUST,2)
        text(str(k),xx,ct+ph+10,21,GRAY,False,'middle')
    d.add(PolyLine(pts,strokeColor=RUST,strokeWidth=3,fillColor=None))
    for r in depth:d.add(Circle(px(int(r['k'])),H-py(float(r['delta'])*100),6,fillColor=RUST,strokeColor=RUST))
    for r in depth:
        if int(r['k']) in [30,50]:
            xx=px(int(r['k']));yy=py(float(r['delta'])*100)
            if int(r['k'])==30:text(f"{float(r['delta'])*100:+.2f} pp",xx+14,yy-43,23,RUST,True)
            else:text(f"{float(r['delta'])*100:+.2f} pp",xx-14,yy+14,23,RUST,True,'end')
    text('Selected minus full recall (pp)',cx,ct-32,21,GRAY)
    text('Recommendation length K',cx+pw/2,ct+ph+41,22,INK,False,'middle')
    para('Intervals: 12 posthoc comparisons; six tested K values. No new fitting or policy selection.',x3,1210,size=21,leading=28,color=GRAY)
    text(f'{100*top10:.2f}%',x3,1290,42,RUST,True)
    para('of excluded true targets were in the original Top10.',x3+190,1291,CW-190,24,31)
    text(f'{100*tail20:.2f}%',x3,1380,42,TEAL,True)
    para('were ranked after position 20 by the full model.',x3+190,1381,CW-190,24,31)

    # Footer provides interpretation, uncertainty scope and provenance.
    line(M,1480,W-M,1480,NAVY,2)
    text('Takeaway',M,1500,27,NAVY,True)
    para('The value of older products depends on scoring and recommendation length. Short-list retention does not make excluded demand unimportant.',M+165,1501,W-2*M-165,25,32,color=INK)
    text('Scope',M,1573,20,NAVY,True)
    text('First-time purchases excluded. Full history retained; no measured runtime/storage gain or causal claim. Seed metrics averaged per user, not scores.',M+82,1573,20,GRAY,width=W-2*M-82)
    text('Evidence',M,1604,18,NAVY,True)
    text('Primary contrasts: user-paired bootstrap, 6,000 draws, simultaneous 95% across 9 contrasts. Posthoc K contrasts use a separate family of 12.',M+92,1604,18,GRAY,width=W-2*M-92)
    text('[1] Instacart (2017), Kaggle / Yasser H. mirror. [2] Ke et al., LightGBM (NeurIPS 2017). Context: Hu et al. (SIGIR 2020); Li et al., Reality Check (2023).',M,1636,17,GRAY,width=W-2*M)
    text('AIGC assisted implementation, analysis and drafting. Team review, authorship details and final wording remain to be completed before submission.',M,1659,17,GRAY,width=W-2*M)

    pdf=OUT/'Poster_Draft.pdf';svg=OUT/'Poster_Draft.svg'
    pdfcanvas=canvas.Canvas(str(pdf),pagesize=(W,H))
    pdfcanvas.setTitle('What should a grocery recommender remember?')
    pdfcanvas.setAuthor('AIAA 3111 team details pending')
    pdfcanvas.setSubject('TEAM REVIEW DRAFT: scoring history and candidate competition')
    renderPDF.draw(d,pdfcanvas,0,0)
    pdfcanvas.showPage();pdfcanvas.save()
    renderSVG.drawToFile(d,str(svg))
    # Reassert a physical A1 viewport while keeping SVG text and vector shapes editable.
    value=svg.read_text(encoding='utf-8')
    import re
    value=re.sub(r'width="[^\"]+" height="[^\"]+"', 'width="841mm" height="594mm"',value,count=1)
    svg.write_text(value,encoding='utf-8')
    (QA/'text_bounds.json').write_text(json.dumps(bounds,indent=2),encoding='utf-8')
    (QA/'poster_manifest.json').write_text(json.dumps({'width_mm':841,'height_mm':594,'pages':1,'vector':True,'text_runs':len(bounds),
         'candidate_reduction':reduction,'excluded_targets':target_total,'excluded_target_top10_share':top10,'excluded_target_beyond20_share':tail20,
         'draft':True,'pdf':str(pdf),'svg':str(svg)},indent=2),encoding='utf-8')
    print(json.dumps({'pdf':str(pdf),'svg':str(svg),'text_runs':len(bounds)}))

if __name__=='__main__':build()
