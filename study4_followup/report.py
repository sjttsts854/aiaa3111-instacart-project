"""Report only the saved posthoc diagnostics; never fit or choose a strategy."""
import json
import shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.data import sha,save_json

NAMES={'full':'完整候选','recency_budget':'同预算最近购买','frequency_budget':'同预算最高频','recent':'仅最近5单'}
EN={'full':'Full pool','recency_budget':'Recency budget','frequency_budget':'Frequency budget','recent':'Recent only'}
COLORS={'full':'#405870','dual':'#157b70','recency_budget':'#b36b26'}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.3f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def save(fig,path):
    fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)

def plot_changes(changes,path):
    d=changes.set_index('comparator').loc[list(NAMES)]
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    y=np.arange(len(d));left=np.zeros(len(d))
    for col,color,label in [('harmed_fraction','#b9553c','Lower recall'),('unchanged_fraction','#d6dce2','Same mean recall'),('improved_fraction','#157b70','Higher recall')]:
        values=d[col].to_numpy()*100;ax[0].barh(y,values,left=left,color=color,label=label);left+=values
    for i,row in enumerate(d.itertuples()):
        ax[0].text(50,i,f'{100*row.unchanged_fraction:.2f}% same; {int(row.improved_users)} higher / {int(row.harmed_users)} lower',ha='center',va='center',fontsize=9)
    ax[0].set(yticks=y,yticklabels=[EN[x] for x in d.index],xlabel='Users with nonempty repeat targets (%)',title='A. Selected policy compared with each control')
    ax[0].invert_yaxis();ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8)
    for i,row in enumerate(d.itertuples()):
        ax[1].barh(i,100*row.gain_contribution,color='#157b70')
        ax[1].barh(i,-100*row.loss_contribution,color='#b9553c')
        ax[1].scatter(100*row.mean_delta,i,color='black',s=28,zorder=4)
    ax[1].axvline(0,color='#666',lw=.8);ax[1].set(yticks=y,yticklabels=[EN[x] for x in d.index],xlabel='Additive population Recall@10 change (pp)',title='B. Gains, losses, and net change (black dot)');ax[1].invert_yaxis()
    fig.suptitle('User-level changes under frozen candidate policies')
    fig.text(.5,-.12,'Posthoc. Three model-seed metrics averaged within each user; equal recall does not imply identical recommendations.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,path)

def plot_subgroups(groups,path):
    dimensions=list(groups.dimension.drop_duplicates())
    fig,axes=plt.subplots(1,len(dimensions),figsize=(15,4.7),squeeze=False)
    for ax,dimension in zip(axes[0],dimensions):
        d=groups[(groups.dimension==dimension)&(groups.comparator=='full')]
        y=np.arange(len(d));x=100*d.mean_delta.to_numpy()
        for pos,row in enumerate(d.itertuples()):ax.plot([100*row.ci95_low,100*row.ci95_high],[pos,pos],color='#157b70',lw=2)
        ax.scatter(x,y,color='#157b70');ax.axvline(0,color='#666',lw=.8)
        ax.set(yticks=y,yticklabels=[f'{r.group} (n={int(r.nonempty_users)})' for r in d.itertuples()],xlabel='Selected minus full Recall@10 (pp)',title=dimension.replace('_',' ').title());ax.invert_yaxis()
    fig.suptitle('Historical user groups: descriptive heterogeneity')
    fig.text(.5,-.04,'Posthoc marginal 95% intervals, not subgroup confirmation. All planned groups are shown; no subgroup is selected for deployment.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,path)

def plot_depth(curves,contrasts,ledger,path):
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for policy in ['full','dual','recency_budget']:
        d=curves[curves.policy.eq(policy)].sort_values('k')
        axes[0].plot(d.k,100*d.recall,'o-',color=COLORS[policy],label='Selected' if policy=='dual' else EN[policy])
    axes[0].set(xlabel='Recommendation list length K',ylabel='Macro recall (%)',title='A. Longer lists retrieve more targets');axes[0].legend(fontsize=8)
    for comparator,color in [('full','#405870'),('recency_budget','#b36b26')]:
        d=contrasts[contrasts.comparator.eq(comparator)].sort_values('k')
        axes[1].errorbar(d.k,100*d.delta,yerr=[100*(d.delta-d.sim_low),100*(d.sim_high-d.delta)],fmt='o-',capsize=3,color=color,label=f'Selected minus {EN[comparator]}')
    axes[1].axhline(0,color='#666',lw=.8);axes[1].axhline(-.5,color='#888',ls='--',lw=1,label='Original -0.5 pp reference')
    axes[1].set(xlabel='K',ylabel='Paired recall change (pp)',title='B. Differences: posthoc simultaneous intervals');axes[1].legend(fontsize=7)
    d=ledger.sort_values('k');x=np.arange(len(d))
    axes[2].bar(x,100*d.promoted_true_recall,color='#157b70',label='Newly promoted true targets')
    axes[2].bar(x,-100*d.deleted_true_recall,color='#b9553c',label='Excluded full-top-K true targets')
    axes[2].plot(x,100*d.delta_recall,'ko-',lw=1,label='Net change')
    axes[2].axhline(0,color='#666',lw=.8);axes[2].set(xticks=x,xticklabels=d.k.astype(int),xlabel='K',ylabel='Additive recall contribution (pp)',title='C. Why pruning changes recall');axes[2].legend(fontsize=7)
    fig.text(.5,-.04,'Frozen scores and policy. Intervals apply only to the six tested K values (12 posthoc contrasts), not a new independent confirmation.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,path)

def plot_target_ranks(ranks,path):
    bands=['1-10','11-20','21-50','>50'];d=ranks.groupby(['eligibility','rank_band']).targets.sum().unstack().reindex(columns=bands,fill_value=0)
    fig,ax=plt.subplots(figsize=(9,4.7));x=np.arange(len(bands));width=.36
    for off,eligibility,color in [(-width/2,'retained','#157b70'),(width/2,'excluded','#b9553c')]:
        values=d.loc[eligibility]/d.loc[eligibility].sum()*100
        ax.bar(x+off,values,width,color=color,label=eligibility.title())
        for p,v in zip(x+off,values):ax.text(p,v+.6,f'{v:.1f}%',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=bands,xlabel='Rank of true repeat targets under the original full model',ylabel='Targets within each eligibility group (%)',title='Which true targets are excluded by candidate pruning?');ax.legend()
    ax.set_ylim(0,max((d.div(d.sum(axis=1),axis=0)*100).max())+9)
    fig.text(.5,-.04,'Posthoc micro target distribution, averaged across three models. Low predicted rank does not establish low user value.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,path)

def report(cache,run,out):
    if not (out/'completed.json').exists():raise RuntimeError('Complete posthoc diagnostics first')
    dest=Path('reports/study4_followup');figs=dest/'figures';tables=dest/'tables';figs.mkdir(parents=True,exist_ok=True);tables.mkdir(exist_ok=True)
    for sub in ['users','depth']:
        for source in (out/sub).glob('*.csv'):
            if source.name not in ['candidate_overlap.csv','top_overlap_seed.csv','top_overlap_users.csv']:
                shutil.copyfile(source,tables/f'{sub}_{source.name}')
    changes=pd.read_csv(out/'users/change_summary.csv');subgroups=pd.read_csv(out/'users/subgroups.csv')
    overlap=pd.read_csv(out/'users/overlap_summary.csv');curves=pd.read_csv(out/'depth/curve_metrics.csv')
    contrasts=pd.read_csv(out/'depth/contrasts.csv');ledger=pd.read_csv(out/'depth/rank_ledger.csv');ranks=pd.read_csv(out/'depth/excluded_target_ranks.csv')
    plot_changes(changes,figs/'01_user_changes.png');plot_subgroups(subgroups,figs/'02_historical_subgroups.png')
    plot_depth(curves,contrasts,ledger,figs/'03_list_depth.png');plot_target_ranks(ranks,figs/'04_target_ranks.png')
    audit=read(out/'audit.json') if (out/'audit.json').exists() else {'status':'PENDING'}
    manifest=read(out/'inputs.json');save_json(dest/'audit_summary.json',audit)
    save_json(dest/'protocol.json',read('study4_followup/protocol.json'))
    save_json(dest/'provenance.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),'posthoc':True,
              'input_manifest_sha256':sha(out/'inputs.json'),'source_freeze_sha256':sha(run/'freeze.json'),
              'source_confirmation_completed':manifest['source_confirmation_completed'],'analysis_completed':read(out/'completed.json')['utc'],
              'audit_status':audit['status'],'table_sha256':{p.name:sha(p) for p in tables.glob('*.csv')}})
    c=changes.set_index('comparator');full=c.loc['full'];recent=c.loc['recency_budget']
    change_table=table(['选定策略减去','改善用户','召回相同','受损用户','总体平均差（百分点）','正向贡献','负向贡献'],
       [[NAMES[r.comparator],pct(r.improved_fraction),pct(r.unchanged_fraction),pct(r.harmed_fraction),pp(r.mean_delta),pp(r.gain_contribution),pp(-r.loss_contribution)] for r in changes.itertuples()])
    depth_table=table(['K','完整候选 Recall','选定策略 Recall','差异（百分点）','事后同时95%区间','被删除真命中贡献','补位真命中贡献'],
       [[int(k),pct(curves[(curves.k==k)&curves.policy.eq('full')].iloc[0].recall),pct(curves[(curves.k==k)&curves.policy.eq('dual')].iloc[0].recall),
         pp(contrasts[(contrasts.k==k)&contrasts.comparator.eq('full')].iloc[0].delta),
         '['+pp(contrasts[(contrasts.k==k)&contrasts.comparator.eq('full')].iloc[0].sim_low)+', '+pp(contrasts[(contrasts.k==k)&contrasts.comparator.eq('full')].iloc[0].sim_high)+']',
         pp(-ledger[ledger.k==k].iloc[0].deleted_true_recall),pp(ledger[ledger.k==k].iloc[0].promoted_true_recall)] for k in sorted(curves.k.unique())])
    rank_counts=ranks.groupby(['eligibility','rank_band']).targets.sum().unstack().fillna(0)
    excluded=rank_counts.loc['excluded'];retained=rank_counts.loc['retained']
    excluded_top10_share=excluded['1-10']/excluded.sum()
    excluded_beyond20_share=(excluded['21-50']+excluded['>50'])/excluded.sum()
    depth_full=contrasts[contrasts.comparator.eq('full')].set_index('k')
    depth_recency=contrasts[contrasts.comparator.eq('recency_budget')].set_index('k')
    rank_table=table(['原完整模型排名','保留目标数（三种子均值）','排除目标数（三种子均值）','占排除目标比例'],
         [[b,f'{retained[b]:.1f}',f'{excluded[b]:.1f}',pct(excluded[b]/excluded.sum())] for b in ['1-10','11-20','21-50','>50']])
    subgroup_table=table(['历史变量','分组','非空目标用户','平均差（百分点）','边际95%区间'],
       [[r.dimension,r.group,int(r.nonempty_users),pp(r.mean_delta),'['+pp(r.ci95_low)+', '+pp(r.ci95_high)+']'] for r in subgroups[subgroups.comparator.eq('full')].itertuples()])
    # Overlap field names are read from the user-analysis schema, below.
    overlap_text=_overlap_narrative(overlap)
    text=f'''# 购物历史应该怎样使用：个体变化、简单对照与推荐长度

本报告接续[第四轮独立确认研究](../study4/research_story.md)，回答平均成绩背后还缺少的三个问题：哪些人的结果变了、同预算的近期选择为何接近双层候选、删去的真实目标在哪里。**这是看过确认结果后的事后分析，不是新一轮独立验证。** 模型、M=5、窗口5和用户均保持不变，没有再次选参或训练。

本轮进一步解释了原结果：**短清单的性能保留主要来自多数用户的推荐未改变，而非大范围收益与损失抵消；被删去的真实目标大多位于原模型排名尾部，随着推荐长度增加，这些目标逐渐进入完整候选方案的前K名，删候选的代价随之显现。** 同预算近期方案的接近表现也有结构原因：两方案都完整保留最近5单商品，只在少量旧商品的选择上不同。这里的“尾部”仅相对当前模型，不能解释为商品不重要。

所有结果沿用完整历史复购目标分母。共8,000名用户，7,472人目标非空；先对每位用户的三个模型指标取均值，再进行比较。这里的“改善/受损”仅指这次离线 Recall 的升降，不是用户满意度或实际购物收益。独立一致性审计：**{audit['status']}**。

## 1. 平均几乎不变，究竟是谁发生了变化？

{change_table}

![逐用户变化与总体得失](figures/01_user_changes.png)

相对完整候选，{pct(full.unchanged_fraction)} 的非空目标用户平均 Recall 相同，{pct(full.improved_fraction)} 提高，{pct(full.harmed_fraction)} 降低。正负贡献都按全部7,472人作分母，二者相加才是总体变化；不能只展示改善者的条件平均。平均绝对变化为 {pp(full.mean_abs_delta)} 个百分点，净变化为 {pp(full.mean_delta)} 个百分点。

具体是 **{int(full.improved_users)} 人改善、{int(full.harmed_users)} 人受损、{int(full.unchanged_users)} 人相同**。变化人数很少，但不能因此说每个人都无损：受损者的条件平均召回下降为 {100*full.harmed_mean_loss:.2f} 个百分点。这是按结果筛选的事后人群，不能当成预测前已知的风险群体，也不应把这种条件均值当成总体损失。

“召回相同”可能意味着清单相同，也可能是不同商品恰好命中相同数量。三个种子的指标平均还可能抵消方向，因此另存了逐种子结果与是否曾在任一种子变化的标记，不把平均为零当作每个模型都完全相同。

## 2. 同预算最近购买策略：相近表现是否来自相近的候选？

{overlap_text}

候选 Jaccard 是交集除以并集；Top10 的有序/无序相同分别要求顺序也一致、或仅商品集合一致。清单相似度本身不是因果证明，但能判断我们究竟比较了多大程度上不同的选择。两个策略使用相同逐用户候选数量和同一评分，差别只在选哪些较早商品。候选差集的历史购买次数与距上次购买订单数见结果表，不能用未来目标来决定保留哪些商品。

## 3. 历史分组能否解释变化发生在哪里？

{subgroup_table}

![历史分组的描述结果](figures/02_historical_subgroups.png)

分组在预测前可计算，但本轮分组分析是在结果已知后制定。所有预先列入本次分析计划的组都报告，区间为2,000次用户 bootstrap 的边际95%区间，没有对大量组间观察作确认性校正。不能据此选择一个“适用人群”，再把同一批用户称为该选择的独立验证。结果还受分组样本量与既有模型限制。高频/近期等候选选择原则的因果或部署收益需要另外的数据验证。

本次相对完整候选的各历史分组区间均包含零，或差异按构造恒为零，未提供稳定的分组获益证据。例如3–5单用户的全部历史本就在最近5单范围内，这一组没有发生候选删减，零差异不能包装成方法有效性的独立发现。旧商品比例高的组也不能仅凭较高点估计就被选为部署对象。

## 4. 推荐清单变长后，损失从哪里出现？

{depth_table}

![清单长度与删除补位账本](figures/03_list_depth.png)

这里固定每个商品原先的完整模型分数，只删候选并让后面的保留商品补位。逐用户恒等式为：**选定方案命中数 − 完整方案命中数 = 新补位真命中 − 被删掉的完整TopK真命中**。两项分别除以同一个完整复购目标数，得到表中可相加的召回贡献。

本次检查 K=5/10/15/20/30/50；两个对照共12项差异使用6,000次用户 bootstrap，同时区间作 Bonferroni 调整，但仍属于事后证据。图中的−0.5个百分点只是原研究预定界限的参考线；不能把本次观察得到的某个K宣称为已独立验证的“安全上限”。K=20点估计下降不自动意味着未通过原界限，也不能用K=10结论保证任意长清单的效果。

实际区间进一步收窄了说法：K=20 的差异为 {pp(depth_full.loc[20,'delta'])} 个百分点，事后同时区间 [{pp(depth_full.loc[20,'sim_low'])}, {pp(depth_full.loc[20,'sim_high'])}]，略跨越−0.5的参考界限，不能在这套事后比较中保证损失小于0.5个百分点；这也不是重新判定原Top10确认实验失败。K=30、50 的差异扩大为 {pp(depth_full.loc[30,'delta'])}、{pp(depth_full.loc[50,'delta'])} 个百分点，区间更明确地显示长清单代价。

另一个有用的边界是：K=30、50 时，选定方案比同预算最近购买分别高 {pp(depth_recency.loc[30,'delta'])}、{pp(depth_recency.loc[50,'delta'])} 个百分点，其事后同时区间在零之上。这提示旧常购选择在更深的列表里可能更有用；但它同时落后完整候选，因此不能把这一局部对照收益包装成总体推荐改进，后续若要提出新策略仍需新数据验证。

## 5. 被排除的真目标并非不存在，它们通常排在哪里？

{rank_table}

![被排除真目标在原模型中的排名](figures/04_target_ranks.png)

排名表是按真实目标商品汇总的 micro 描述，和用户平均 Recall 的口径不同。它衡量既有模型把目标排在哪里，不能推断低位商品不重要、需求消失、用户已经遗忘，或任何更好的模型也预测不了。候选删减制造的不可召回目标仍被计入分母。排名尾部的目标在短清单中原本很少被命中，长清单或更强模型则可能改变这笔取舍。

被排除的 **{int(round(excluded.sum())):,} 个真实复购目标**中，仅 **{pct(excluded_top10_share)}** 原先位于完整模型前10名，**{pct(excluded_beyond20_share)}** 位于20名以后（排名按三个模型汇总平均）。这解释了为何目标覆盖率已经下降，短清单召回却几乎不变；也解释了为何推荐变长后，原先隐藏在后排的删除代价开始出现。这里描述的是原固定模型中的命中去向，并非证明被删商品可以永久舍弃。

## 6. 更新后的主线与贡献定位

建议题目为 **《购物推荐应该记住什么？——历史信息、候选竞争与选择性保留》**。课堂讲述从长短历史用户的原始差距引入，再用第四轮的同用户、同目标对照区分评分信息与候选竞争；本报告补充个体得失、简单对照相似度与清单长度范围。

可保留的贡献是：固定评价对象的实验设计、旧商品找回与挤出的机制证据、候选精简的性能与覆盖取舍。不能把同预算近期策略的接近表现藏起来，也不能把事后分组或新的K曲线包装成新的方法胜利。45%左右候选总量下降仍不等于实测评分提速，完整历史特征和原始全量分数一直保留。

复现：`python -m study4_followup.run audit` 与 `python -m study4_followup.run report`。分析计划与输入时序见 [protocol.json](protocol.json)、[provenance.json](provenance.json)，精选结果在 [tables/users_change_summary.csv](tables/users_change_summary.csv) 与 [tables/depth_contrasts.csv](tables/depth_contrasts.csv)。原第四轮确认结论保留不变；这里新增的是解释和适用范围。
'''
    (dest/'analysis.md').write_text(text,encoding='utf-8')
    return {'report':str(dest/'analysis.md'),'figures':4,'posthoc':True,'audit':audit['status']}

def _overlap_narrative(overlap):
    all_users=overlap[overlap.population.eq('all')].set_index('comparator')
    rows=[]
    for comparator in ['full','recency_budget','frequency_budget','recent']:
        row=all_users.loc[comparator]
        rows.append([NAMES[comparator],int(row.users),pct(row.top_ordered_identical_seedmean),
                     pct(row.top_all_seeds_ordered_identical_fraction),pct(row.top_mean_jaccard)])
    tab=table(['对照','全部用户','逐模型清单完全相同比例的均值','三个模型清单均完全相同','Top10平均Jaccard'],rows)
    row=all_users.loc['recency_budget']
    return tab+f'''\n\n以上使用全部8,000用户，清单是否相同无需用户有复购目标；只取7,472名非空目标用户的对应结果另存于表中。相对完整候选，{pct(all_users.loc['full','top_all_seeds_ordered_identical_fraction'])} 的用户在三个模型下都保持原清单。表中“逐模型均值”与“所有模型都相同”是不同统计量。固定同一分数与稳定排序时，两个清单集合相同则顺序也相同；有序与无序统计在本数据中一致。

同预算近期方案与选定方案的**整体候选平均Jaccard为 {pct(row.candidate_mean_jaccard)}**，但仅比较两边出现过的旧商品，平均Jaccard降为 **{pct(row.candidate_old_mean_jaccard)}**（只对旧商品并集非空的{int(row.candidate_old_union_nonempty_users):,}人计算）。所有用户的近期商品都被两种方案共同保留；高整体重合率因此部分来自设计，而不是两种旧商品选择真的完全等价。三个模型都给出完全相同Top10的用户占 **{pct(row.top_all_seeds_ordered_identical_fraction)}**。

这为“简单近期对照为何接近”提供了当前固定排名下的解释：两方案的大部分候选相同，剩余差异又很少改变前10名。不能据此推出长期频次信息在所有推荐长度、模型或数据集上都没有价值。'''
