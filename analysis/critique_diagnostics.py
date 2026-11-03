"""Post-evaluation diagnostics for critique, not model selection.
Run: python3 analysis/critique_diagnostics.py
Does not fit or alter any predictive model.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score
out=Path('analysis/results')
s=pd.read_csv(out/'splits.csv');d=pd.read_csv('data/add.csv',index_col=0,low_memory=False)
x=d.iloc[:,:-1].apply(lambda a:pd.to_numeric(a.astype(str).str.strip(),errors='coerce'))
tr=s.partition.eq('train').to_numpy();names=[l.split(':')[0] for l in Path('analysis/source/ad.names').read_text(encoding='latin1').splitlines() if ':' in l and not l.startswith('|')]
x.columns=names
p=pd.read_csv(out/'test_predictions.csv');v=pd.read_csv(out/'validation_predictions.csv');threshold=float(pd.read_csv(out/'summary.csv').set_index('item').loc['threshold','value'])
info={}
for part,pr in [('validation',v),('test',p)]:
 score=pr.full_ridge;pred=score.ge(threshold);groups=s.set_index('row').loc[pr.row,'group'].to_numpy();a=pr.assign(group=groups,pred=pred,error=pred!=pr.y)
 err=a.groupby('group').agg(n=('y','size'),errors=('error','sum'),y=('y','first'),score=('full_ridge','first')).query('errors>0').sort_values('errors',ascending=False)
 err.to_csv(out/f'critique_{part}_error_groups.csv')
 rows=[]
 for t in [0.1,threshold,0.2,0.3,0.5]:
  h=score.ge(t);rows.append(dict(threshold=t,precision=precision_score(pr.y,h,zero_division=0),recall=recall_score(pr.y,h),F1=f1_score(pr.y,h),positive_predictions=int(h.sum())))
 info[part]={'error_rows':int(a.error.sum()),'error_groups':len(err),'largest_error_groups':err.head(5).reset_index().to_dict('records'),'threshold_summary':rows}
 if part=='validation':
  cuts=np.unique(np.r_[0,score,1]);fs=np.array([f1_score(pr.y,score.ge(t)) for t in cuts]);best=cuts[fs==fs.max()];info[part]['maximizing_candidate_count']=len(best);info[part]['maximizing_candidates']=best.tolist()
  decisions=score.ge(threshold);info[part]['same_decision_interval']={'lower_open':float(score[~decisions].max()),'upper_closed':float(score[decisions].min())}
# A post-hoc failure slice, not a new model-selection criterion.
subgroups=[]
for part,pr in [('validation',v),('test',p)]:
    xx=x.iloc[pr.row.to_numpy()-1,:3]
    missing=xx.isna().any(axis=1).to_numpy()
    for value in [False,True]:
        q=pr.loc[missing==value];h=q.full_ridge.ge(threshold)
        subgroups.append(dict(partition=part,missing_geometry=value,n=len(q),ads=int(q.y.sum()),TP=int((h*q.y).sum()),FP=int((h*(1-q.y)).sum()),F1=f1_score(q.y,h)))
pd.DataFrame(subgroups).to_csv(out/'critique_geometry_subgroups.csv',index=False)
# Development evidence, not a new test gate.
cv=pd.read_csv(out/'cv.csv');means=cv.groupby(['model','parameter']).AP.mean();rows=[]
for model in cv.model.unique():
 m=means.loc[model];best=m.idxmax();a=cv[(cv.model==model)&np.isclose(cv.parameter,best)];rows.append(dict(model=model,parameter=best,mean=a.AP.mean(),sd=a.AP.std(),min=a.AP.min(),max=a.AP.max(),fold_values=a.AP.tolist(),near_best_candidates=int((m>=m.max()-.01).sum())))
info['cv_selected']=rows
binary=x.iloc[:,3:];count=binary.iloc[np.where(tr)[0]].sum();never=count.eq(0);rare=count.le(5)
info['binary_support']={'never_active_training':int(never.sum()),'active_at_most_five_training':int(rare.sum()),'all_binary':binary.shape[1]}
for part in ['validation','test']:
 xx=binary.loc[s.partition.eq(part)];info['binary_support'][part+'_rows_with_unseen_active_feature']=int(xx.loc[:,never].eq(1).any(axis=1).sum())
# Recover class-specific descriptive evidence for the context families.
families=[]
for fam in ['url','origurl','ancurl','alt','caption']:
 cols=[n for n in names if n.startswith(fam+'*')];sums=x[cols].sum(axis=1)
 for y in [0,1]:
  z=sums[s.y.eq(y).to_numpy()];families.append(dict(family=fam,features=len(cols),label=y,mean_active=z.mean(),median_active=z.median(),zero_active_fraction=z.eq(0).mean()))
pd.DataFrame(families).to_csv(out/'critique_context_summary.csv',index=False)
# Whether geometry has deterministic structure matters more than a generic correlation matrix.
obs=x.iloc[:,:3].notna().all(axis=1);delta=(x.loc[obs,'aratio']-x.loc[obs,'width']/x.loc[obs,'height']).abs();info['geometry_identity']={'complete_rows':int(obs.sum()),'within_0_00011':int(delta.le(.00011).sum()),'max_absolute_difference':float(delta.max())}
info['all_missing_training_fold_features_first_four']=sum(int(x.iloc[np.where(tr & s.fold.ne(k).to_numpy())[0],:4].isna().all().sum()) for k in range(1,6))
(out/'critique_diagnostics.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
