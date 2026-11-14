"""Sparse TIFU-KNN adapted to the study's repeat-only evaluation protocol.

Temporal aggregation follows the author's released TIFUKNN.py, which differs
from the paper's prose for within-group decay and unequal group allocation.
See tifu_sources.md. No targets enter this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


def validate_settings(settings):
    if settings.get("implementation","author_code")!="author_code":
        raise ValueError("Only explicitly documented author_code implementation is supported")
    m=settings["group_count"]
    if isinstance(m,bool) or int(m)!=m or m<1:raise ValueError("group_count must be a positive integer")
    if not all(0<float(settings[k])<=1 for k in ["within_decay","group_decay"]):
        raise ValueError("Decay rates must be in (0,1]")


def order_weights(n_orders, settings):
    """Weight of each chronological binary basket in the released author code.

    group_count means NUMBER of groups, not baskets per group. Remainder
    baskets go to the recent groups. Short histories have one basket per
    group but retain the nominal group_count in the across-group exponent.
    """
    validate_settings(settings)
    if int(n_orders)!=n_orders or n_orders<1:raise ValueError("Empty/noninteger history")
    n=int(n_orders);m=int(settings["group_count"]);g=min(n,m)
    sizes=np.full(g,n//g,dtype=np.int64)
    if n%g:sizes[-(n%g):]+=1
    group=np.repeat(np.arange(g),sizes)
    within=float(settings["within_decay"])**np.arange(n-1,-1,-1)
    across=float(settings["group_decay"])**(m-1-group)
    return within*across/(sizes[group]*g)


def build_vectors(history, settings, n_items=None):
    """Return sorted user ids and CSR vectors indexed by raw integer product id.

    All catalog items are retained when n_items=max_catalog_product_id+1.
    Histories must start at order 1 and be contiguous; input baskets are sets.
    """
    validate_settings(settings)
    required=["user_id","order_number","product_id"]
    if not set(required).issubset(history):raise ValueError("Missing history columns")
    if history.empty:raise ValueError("History is empty")
    if "eval_set" in history and not history.eval_set.eq("prior").all():
        raise ValueError("TIFU vectors accept prior history only")
    h=history[required]
    if h.isna().any().any():raise ValueError("Missing history identifiers")
    raw=h.to_numpy()
    if not np.equal(raw,np.floor(raw)).all():raise ValueError("Identifiers must be integers")
    if (h.order_number<1).any() or (h.product_id<0).any():raise ValueError("Invalid order or product identifier")
    if h.duplicated(required).any():raise ValueError("A basket must contain a product at most once")
    count=h.groupby("user_id").order_number.nunique().sort_index()
    maximum=h.groupby("user_id").order_number.max().reindex(count.index)
    if not count.eq(maximum).all():raise ValueError("History must contain contiguous orders beginning at 1")
    users=count.index.to_numpy(dtype=np.int64)
    n_items=int(h.product_id.max())+1 if n_items is None else int(n_items)
    if n_items<=int(h.product_id.max()):raise ValueError("Catalog dimension excludes observed products")
    nmax=int(maximum.max());lookup=np.zeros((nmax+1,nmax))
    for n in count.unique():lookup[int(n),:int(n)]=order_weights(int(n),settings)
    rows=pd.Index(users).get_indexer(h.user_id)
    lengths=count.to_numpy(dtype=np.int64)[rows]
    weights=lookup[lengths,h.order_number.to_numpy(dtype=np.int64)-1]
    matrix=coo_matrix((weights,(rows,h.product_id.to_numpy(dtype=np.int64))),shape=(len(users),n_items)).tocsr()
    matrix.sum_duplicates();matrix.sort_indices()
    return users,matrix


def fit(history, train_user_ids, settings, n_items=None):
    """Build a neighbor bank exclusively from the declared training ids."""
    ids=np.sort(np.unique(np.asarray(train_user_ids,dtype=np.int64)))
    if not len(ids):raise ValueError("Training user list is empty")
    if n_items is None:n_items=int(history.product_id.max())+1
    subset=history[history.user_id.isin(ids)]
    users,matrix=build_vectors(subset,settings,n_items)
    if not np.array_equal(users,ids):raise ValueError("Some training users have no history")
    return {"settings":dict(settings),"train_user_ids":users,"matrix":matrix,"n_items":n_items,
            "squared_norms":np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel(),
            "scope":"training-history neighbors; repeat-only candidate adaptation"}


def _nearest_batch(query, query_users, artifact, k):
    """Exact Euclidean search; deterministic ties by ascending training user id."""
    bank=artifact["matrix"]
    query_norm=np.asarray(query.multiply(query).sum(axis=1)).ravel()
    distances=(query@bank.T).toarray()
    distances*=-2
    distances+=query_norm[:,None]
    distances+=artifact["squared_norms"][None,:]
    np.maximum(distances,0,out=distances)
    self_index=pd.Index(artifact["train_user_ids"]).get_indexer(query_users)
    present=np.flatnonzero(self_index>=0)
    distances[present,self_index[present]]=np.inf
    if k>bank.shape[0]-int(bool(len(present))):raise ValueError("Too many neighbors after excluding self")
    # Partial selection plus boundary-tie repair avoids full 20k sorting per row.
    partition=np.argpartition(distances,k-1,axis=1)[:,:k]
    out=np.empty_like(partition)
    for i,chosen in enumerate(partition):
        cutoff=distances[i,chosen].max()
        below=np.flatnonzero(distances[i]<cutoff)
        tied=np.flatnonzero(distances[i]==cutoff)[:k-len(below)]
        chosen=np.r_[below,tied]
        out[i]=chosen[np.lexsort((artifact["train_user_ids"][chosen],distances[i,chosen]))]
    return out,np.sqrt(np.take_along_axis(distances,out,axis=1))


def find_neighbors(query_matrix,query_users,artifact,k,batch_size=128):
    """Diagnostic/search API. Returns training ROW indices and distances."""
    if int(k)!=k or k<1:raise ValueError("k must be positive")
    if int(batch_size)!=batch_size or batch_size<1:raise ValueError("Invalid batch size")
    if len(query_users)!=query_matrix.shape[0]:raise ValueError("Query alignment mismatch")
    parts=[];distances=[]
    for start in range(0,len(query_users),batch_size):
        idx,dist=_nearest_batch(query_matrix[start:start+batch_size],query_users[start:start+batch_size],artifact,int(k))
        parts.append(idx);distances.append(dist)
    return np.vstack(parts),np.vstack(distances)


def components(query_history,candidates,artifact,neighbor_counts=(50,300,900),batch_size=128):
    """Compute personal and neighbor scores once for a bounded alpha/k grid.

    Every output array follows the input candidate row order exactly. Query
    batches bound dense distance memory to batch_size * training_users; no
    dense users-by-catalog matrix is constructed. Neighbor means are sparse.
    """
    if int(batch_size)!=batch_size or batch_size<1:raise ValueError("Invalid batch size")
    counts=sorted(set(int(k) for k in neighbor_counts))
    if not counts or counts[0]<1:raise ValueError("Positive neighbor counts required")
    if candidates.duplicated(["user_id","product_id"]).any():raise ValueError("Duplicate candidates")
    users,query=build_vectors(query_history,artifact["settings"],artifact["n_items"])
    row=pd.Index(users).get_indexer(candidates.user_id)
    column=candidates.product_id.to_numpy(dtype=np.int64)
    if (row<0).any() or (column<0).any() or (column>=artifact["n_items"]).any():
        raise ValueError("Candidate lacks history or is outside catalog")
    personal=np.asarray(query[row,column]).ravel()
    if (personal<=0).any():raise ValueError("Repeat-only candidates must have personal history")
    scores={k:np.empty(len(candidates),dtype=np.float64) for k in counts}
    for start in range(0,len(users),batch_size):
        stop=min(start+batch_size,len(users));n=stop-start
        idx,_=_nearest_batch(query[start:stop],users[start:stop],artifact,counts[-1])
        selected=np.flatnonzero((row>=start)&(row<stop));local=row[selected]-start
        for k in counts:
            aggregation=csr_matrix((np.full(n*k,1/k),
                (np.repeat(np.arange(n),k),idx[:,:k].ravel())),shape=(n,len(artifact["train_user_ids"])))
            mean=aggregation@artifact["matrix"]
            scores[k][selected]=np.asarray(mean[local,column[selected]]).ravel()
    return {"personal":personal,"neighbors":scores,"diagnostics":{
        "training_users":len(artifact["train_user_ids"]),"query_users":len(users),"candidate_rows":len(candidates),
        "batch_size":batch_size,"neighbor_counts":counts,"metric":"Euclidean","self_excluded":True,
        "implementation":"author_code","dense_distance_shape_bound":[batch_size,len(artifact["train_user_ids"])]}}


def combine(parts,k,alpha):
    """Alpha=1 is the temporal-personal-only ablation, not collaborative TIFU."""
    if not 0<=alpha<=1:raise ValueError("alpha must lie in [0,1]")
    return alpha*parts["personal"]+(1-alpha)*parts["neighbors"][k]
