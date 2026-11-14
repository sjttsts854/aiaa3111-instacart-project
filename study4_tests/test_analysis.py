import numpy as np
import pandas as pd
import pytest
from study4.analysis import linear_contrast

def test_interaction_pairs_by_user_and_preserves_empty_targets():
    aa=pd.DataFrame({'user_id':[1,2,3],'recall':[.6,0,np.nan]})
    ra=pd.DataFrame({'user_id':[2,3,1],'recall':[.1,np.nan,.4]})
    ar=pd.DataFrame({'user_id':[1,2,3],'recall':[.5,.1,np.nan]})
    rr=pd.DataFrame({'user_id':[1,2,3],'recall':[.4,0,np.nan]})
    result=linear_contrast([aa,ra,ar,rr],[1,-1,-1,1])
    np.testing.assert_allclose(result.recall,[.1,-.2,np.nan],equal_nan=True)
    with pytest.raises(ValueError,match='universes'):
        linear_contrast([aa,ra.iloc[:2]],[1,-1])
    changed=ra.copy();changed.loc[changed.user_id.eq(3),'recall']=0
    with pytest.raises(ValueError,match='denominators'):
        linear_contrast([aa,changed],[1,-1])
