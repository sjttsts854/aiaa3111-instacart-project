"""Exercise the portable English viewer with actual frozen evidence."""
import argparse,json
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

def check(root):
    app=AppTest.from_file(str(root/'evidence_app.py'),default_timeout=60).run()
    assert not app.exception,[e.message for e in app.exception]
    assert app.radio[0].options==['Study and results','Individual case','Posthoc scope']
    app.radio[0].set_value('Individual case').run()
    assert not app.exception,[e.message for e in app.exception]
    assert len(app.selectbox[0].options)==8000
    targets=pd.read_parquet(root/'data_demo/targets.parquet')
    ids=pd.read_parquet(root/'data_demo/users.parquet').user_id
    repeat=targets[targets.is_repeat].groupby('user_id').size().reindex(ids,fill_value=0)
    ordinary=int(repeat[repeat.gt(0)].index[0]);empty=int(repeat[repeat.eq(0)].index[0])
    meta=json.loads((root/'data_demo/metadata.json').read_text(encoding='utf-8'))
    # Ordinary case also demonstrates a mixed repeat/new-product basket.
    scope=targets[targets.user_id.eq(ordinary)]
    assert scope.is_repeat.any() and (~scope.is_repeat).any()
    for user in [ordinary,meta['suggested_changed_list_user'],meta['suggested_improved_user'],meta['suggested_harmed_user'],empty]:
        app.selectbox[0].set_value(user).run();app.checkbox[0].check().run()
        assert not app.exception,[e.message for e in app.exception]
        assert app.radio[0].value=='Individual case'
        assert app.header[0].value=='Inspect a genuine Study 4 confirmation case'
        assert any(f'Repeat targets: {int(repeat.loc[user])};' in e.value for e in app.markdown)
        for table in app.dataframe[-3:]:
            assert list(table.value.columns)==['rank','label','product_name']
            assert table.value.label.dtype==bool
    assert any('No repeat targets' in e.value for e in app.info)
    for view,heading in [('Posthoc scope','Posthoc interpretation and limits'),
                         ('Study and results','Separate scoring history from candidate eligibility')]:
        app.radio[0].set_value(view).run()
        assert not app.exception,[e.message for e in app.exception]
        assert app.header[0].value==heading
    return {'status':'PASS','users':8000,'ordinary_case':ordinary,'empty_case':empty,
            'changed_list_case':meta['suggested_changed_list_user'],'improved_case':meta['suggested_improved_user'],
            'harmed_case':meta['suggested_harmed_user'],'scope_case_contains_repeat_and_new_products':True,
            'views':3,'navigation_persists_after_case_changes':True,'hit_column_precedes_product':True,'single_case_seed':42}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    print(json.dumps(check(parser.parse_args().root),indent=2))
