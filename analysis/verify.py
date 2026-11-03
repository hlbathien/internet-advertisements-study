"""Independent checks of exported R predictions and source identity.
Run from project root: python3 analysis/verify.py
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
root=Path('analysis/results')
d=pd.read_csv('data/add.csv',index_col=0,low_memory=False)
u=pd.read_csv('analysis/source/ad.data',header=None,low_memory=False)
x=d.iloc[:,:-1].apply(lambda c:pd.to_numeric(c.astype(str).str.strip(),errors='coerce'))
ux=u.iloc[:,:-1].apply(lambda c:pd.to_numeric(c.astype(str).str.strip(),errors='coerce'))
assert np.allclose(x,ux,equal_nan=True)
assert np.array_equal(d.iloc[:,-1],u.iloc[:,-1])
s=pd.read_csv(root/'splits.csv');p=pd.read_csv(root/'test_predictions.csv');r=pd.read_csv(root/'test_metrics.csv')
assert len(s)==len(d) and s.row.is_unique
# Independently recompute duplicate groups, so a faulty grouping column cannot pass.
h=pd.util.hash_pandas_object(x,index=False)
assert pd.DataFrame({'h':h.values,'p':s.partition.values}).groupby('h').p.nunique().max()==1
train=s.partition.eq('train').to_numpy()
assert pd.DataFrame({'h':h.values[train],'fold':s.fold.values[train]}).groupby('h').fold.nunique().max()==1
assert np.array_equal(p.row,s.loc[s.partition=='test','row'])
assert np.array_equal(p.y,(d.iloc[p.row.to_numpy()-1,-1]=='ad.').astype(int))
for row in r.itertuples():
    score=p['selected' if row.model=='selected_tuned' else row.model] if row.model!='always_nonad' else np.repeat(s.loc[s.partition=='train','y'].mean(),len(p))
    pred=(score>=row.threshold).astype(int) if row.model!='always_nonad' else np.zeros(len(p),dtype=int)
    yy=p.y
    checks={'accuracy':accuracy_score(yy,pred),'balanced_accuracy':balanced_accuracy_score(yy,pred),'precision':precision_score(yy,pred,zero_division=0),'recall':recall_score(yy,pred),'F1':f1_score(yy,pred),'ROC_AUC':roc_auc_score(yy,score),'AP':average_precision_score(yy,score),'Brier':brier_score_loss(yy,score)}
    tn,fp,fn,tp=confusion_matrix(yy,pred).ravel()
    checks.update(TP=tp,FP=fp,FN=fn,TN=tn,specificity=tn/(tn+fp))
    for k,v in checks.items(): assert np.isclose(v,getattr(row,k),rtol=1e-8), (row.model,k,v,getattr(row,k))
v=pd.read_csv(root/'validation_predictions.csv');summary=pd.read_csv(root/'summary.csv').set_index('item').value
threshold=float(summary['threshold']);sel=summary['selected_model']
thresholds=np.unique(np.r_[0,v[sel],1]);vals=np.array([f1_score(v.y,v[sel]>=t) for t in thresholds]);assert np.isclose(threshold,thresholds[vals==vals.max()].max())
cv=pd.read_csv(root/'cv_summary.csv');assert cv.loc[cv.AP.idxmax(),'model']==sel
assert np.array_equal(p.decision,(p.selected>=threshold).astype(int))
family={}
for line in Path('analysis/source/ad.names').read_text(encoding='latin-1').splitlines():
    if ':' in line and not line.startswith('|'):
        name=line.split(':')[0]; f=name.split('*')[0] if '*' in name else name;family[f]=family.get(f,0)+1
q=p.drop_duplicates('group')
group_metrics={'AP':average_precision_score(q.y,q.selected),'ROC_AUC':roc_auc_score(q.y,q.selected),'F1':f1_score(q.y,q.decision),'n':len(q)}
counts=pd.DataFrame({'g':s.group,'y':s.y}).groupby('g').y.agg(['count','sum']);conf=counts[(counts['sum']>0)&(counts['sum']<counts['count'])]
info={'source_exact_match':True,'source_sha256':hashlib.sha256(Path('data/add.csv').read_bytes()).hexdigest(),'missing_any':int(x.isna().any(axis=1).sum()),'missing_any_geometry':int(x.iloc[:,:3].isna().any(axis=1).sum()),'binary_density':float(x.iloc[:,3:].stack().mean()),'families':family,'contradictory_group_sizes':conf.to_dict('index'),'irreducible_in_sample_errors_from_identical_features':int(np.minimum(conf['sum'],conf['count']-conf['sum']).sum()),'validation_selected_F1':float(vals.max()),'test_one_representative_per_group':group_metrics,'verified_models':len(r)}
(root/'independent_verification.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
