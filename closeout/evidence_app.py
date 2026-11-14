"""Portable, read-only English viewer for the frozen Study 4 evidence."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data_demo'
EVIDENCE=ROOT/'evidence'
st.set_page_config(page_title='Grocery Memory Study',page_icon='🛒',layout='wide')
st.title('What should a grocery recommender remember?')
st.caption('AIAA 3111 | Team review draft | Frozen offline evidence, not live model training')

@st.cache_data
def load():
    return {name:pd.read_parquet(DATA/f'{name}.parquet') for name in ['users','historical_catalog','targets','predictions','products']}

data=load()
names={'full':'Full historical pool','dual':'Recent + old frequent','recency_budget':'Same-budget recency'}
active_view=st.radio('View',['Study and results','Individual case','Posthoc scope'],horizontal=True,key='active_view')
if active_view=='Study and results':
    st.header('Separate scoring history from candidate eligibility')
    st.write('The task predicts historically purchased products in a user’s next recorded basket. Candidate pruning never removes an actual repeat target from evaluation. New products remain outside the modeled candidate scope.')
    st.write('Study 4 used 20,000 training, 5,000 validation and 8,000 fresh confirmation users. Of the confirmation users, 7,472 had at least one repeat target. Models, policy selection and evaluation were frozen before confirmation.')
    metrics=pd.read_csv(EVIDENCE/'study4/tables/confirmation_metrics.csv').set_index('name')
    for column,name in zip(st.columns(3),names):
        column.metric(f'{names[name]} Recall@10',f"{metrics.loc[name,'recall_at_10']:.3%}")
        column.caption(f"Mean eligible candidates: {metrics.loc[name,'mean_candidates']:.2f}")
    st.caption('Study-level metrics average per-user metrics across three fitted models; they are not predictions from an ensemble.')
    contrast=pd.read_csv(EVIDENCE/'study4/tables/primary_contrasts.csv').set_index('contrast').loc['dual_vs_full']
    st.write(f"Selected minus full recall: {100*contrast.delta:+.4f} percentage points; simultaneous 95% interval [{100*contrast.sim_low:+.4f}, {100*contrast.sim_high:+.4f}]. It passes the prespecified -0.5 percentage-point noninferiority limit, without establishing higher accuracy.")
    st.image(str(EVIDENCE/'study4/figures/02_factorial_memory.png'))
    st.write('For the same long-history users and full targets, expanding the historical candidate pool hurts frequency scoring but helps time-decayed scoring. The meaning of old history depends on how it is used.')
    st.image(str(EVIDENCE/'study4/figures/03_primary_contrasts.png'))
    st.warning('Candidate count reduction is not measured runtime or storage savings. Historical features and full-pool scores were retained. This study does not infer forgotten shopping needs or causal changes in consumer behavior.')
elif active_view=='Individual case':
    st.header('Inspect a genuine Study 4 confirmation case')
    st.caption('Every case uses the same frozen seed-42 model, with three candidate policies. The case results are a single model, whereas headline tables average three models. Users are anonymous dataset IDs.')
    meta=json.loads((DATA/'metadata.json').read_text(encoding='utf-8'))
    st.caption(f"Illustrative IDs: {meta['suggested_ordinary_user']} (repeat/new-product scope), {meta['suggested_changed_list_user']} (changed list), {meta['suggested_improved_user']} (more hits), {meta['suggested_harmed_user']} (fewer hits), {meta['suggested_empty_user']} (no repeat target). Each is the smallest eligible ID for its condition, not a representative sample.")
    users=data['users'].set_index('user_id')
    chosen=st.selectbox('Confirmation user',users.index.astype(int).sort_values().tolist())
    user=users.loc[chosen]
    catalog=data['historical_catalog'].query('user_id == @chosen')
    target=data['targets'].query('user_id == @chosen')
    products=data['products'].set_index('product_id')
    cols=st.columns(3)
    cols[0].metric('Observed prior orders',int(user.history_orders))
    cols[1].metric('Full historical candidates',len(catalog))
    cols[2].metric('Selected eligible candidates',int(catalog.dual_eligible.sum()))
    with st.expander('Historical purchase information'):
        history=catalog.merge(products[['product_name']],left_on='product_id',right_index=True)
        st.dataframe(history.sort_values(['purchase_count','orders_since_purchase','product_id'],ascending=[False,True,True])[
            ['product_name','purchase_count','orders_since_purchase','dual_eligible','recency_eligible']].rename(columns={
                'product_name':'Product','purchase_count':'Past purchases','orders_since_purchase':'Orders since last purchase',
                'dual_eligible':'Eligible in selected policy','recency_eligible':'Eligible in recency control'}),hide_index=True)
    reveal=st.checkbox('Reveal the recorded next basket and evaluation',value=False)
    repeat_count=int(target.is_repeat.sum())
    if reveal:
        st.write(f'Repeat targets: {repeat_count}; whole-basket products: {len(target)}. These denominators stay fixed for every policy.')
        if repeat_count==0:st.info('No repeat targets: Recall@10 is undefined for this user. The user remains in the case viewer and in other applicable metrics.')
        truth=target.merge(products[['product_name']],left_on='product_id',right_index=True)
        st.dataframe(truth[['product_name','is_repeat']].rename(columns={'product_name':'Recorded next-basket product','is_repeat':'Seen in prior history'}),hide_index=True)
    for column,(name,label) in zip(st.columns(3),names.items()):
        with column:
            top=data['predictions'].query('user_id == @chosen and policy == @name').sort_values('rank')
            view=top.merge(products[['product_name']],left_on='product_id',right_index=True)
            st.subheader(label)
            fields=['rank']+(['label'] if reveal else [])+['product_name']
            if reveal:view['label']=view.label.astype(bool)
            st.dataframe(view[fields],hide_index=True,use_container_width=True,column_config={
                'rank':st.column_config.NumberColumn('Rank',width='small',format='%d'),
                'label':st.column_config.CheckboxColumn('Hit',width='small',help='Product appears in the recorded next basket as a repeat purchase'),
                'product_name':st.column_config.TextColumn('Product',width='medium')})
            if reveal:
                hits=int(top.label.sum())
                st.write(f'Hits: {hits}/{repeat_count}; Recall@10: '+(f'{hits/repeat_count:.1%}' if repeat_count else 'undefined'))
                st.caption(f'Whole-basket recall: {hits/len(target):.1%}; returned {len(top)} products.')
elif active_view=='Posthoc scope':
    st.header('Posthoc interpretation and limits')
    st.warning('These analyses were designed after Study 4 confirmation results were inspected. They do not constitute a new independent confirmation, and they did not train models or select a new candidate policy.')
    changes=pd.read_csv(EVIDENCE/'study4_followup/tables/users_change_summary.csv').set_index('comparator').loc['full']
    st.write(f"Among {int(changes.nonempty_users):,} users with repeat targets, {int(changes.unchanged_users):,} had unchanged mean recall, {int(changes.improved_users)} improved, and {int(changes.harmed_users)} declined.")
    st.image(str(EVIDENCE/'study4_followup/figures/01_user_changes.png'))
    st.image(str(EVIDENCE/'study4_followup/figures/03_list_depth.png'))
    st.write('Only 0.26% of excluded true repeat targets originally ranked in the full model’s Top10; 95.54% ranked after position 20. Lower predicted rank does not mean lower user value. Longer lists expose coverage losses that short-list metrics can hide.')
    st.image(str(EVIDENCE/'study4_followup/figures/04_target_ranks.png'))
    st.caption('K curves use posthoc simultaneous intervals over 12 comparisons. The original Top10 confirmation used a separate prespecified family of 9 comparisons. Neither supports selecting a new safe K on these same users.')
