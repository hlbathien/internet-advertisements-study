"""Describe fixed test probabilities; do not fit a calibration model."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=pd.read_csv('analysis/results/test_predictions.csv')
bins=np.linspace(0,1,6)
a=p.assign(bin=pd.cut(p.selected,bins,include_lowest=True)).groupby('bin',observed=True).agg(n=('y','size'),predicted=('selected','mean'),observed=('y','mean'))
a.to_csv('analysis/results/reliability.csv')
fig,ax=plt.subplots(1,2,figsize=(8,3.3))
ax[0].plot([0,1],[0,1],'--',color='gray');ax[0].plot(a.predicted,a.observed,'o-',color='#2166ac')
for r in a.itertuples():ax[0].annotate(f'n={r.n}',(r.predicted,r.observed),xytext=(3,-13),textcoords='offset points',fontsize=8)
ax[0].set(xlim=(0,1),ylim=(0,1.05),xlabel='Mean predicted probability',ylabel='Observed ad fraction',title='Coarse reliability display')
ax[1].hist([p.loc[p.y==0,'selected'],p.loc[p.y==1,'selected']],bins=np.linspace(0,1,11),label=['nonad','ad'],color=['#a6cee3','#fb9a99']);ax[1].set(xlabel='Predicted ad probability',ylabel='Records',title='Test score distribution');ax[1].legend()
fig.tight_layout();fig.savefig('analysis/results/reliability.pdf');plt.close(fig)

# Compare the frozen models; no additional fitting or threshold selection.
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, f1_score
fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
for model, label in [('geometry_ridge', 'Geometry ridge'), ('full_ridge', 'Full ridge'), ('tree', 'Tree')]:
    scores = p[model]
    fpr, tpr, _ = roc_curve(p.y, scores)
    precision, recall, _ = precision_recall_curve(p.y, scores)
    auc = roc_auc_score(p.y, scores)
    ap = average_precision_score(p.y, scores)
    ax[0].plot(fpr, tpr, label=f'{label}: {auc:.3f}')
    ax[1].step(recall[::-1], precision[::-1], where='pre', label=f'{label}: {ap:.3f}')
ax[0].plot([0, 1], [0, 1], '--', color='gray')
ax[1].axhline(p.y.mean(), linestyle='--', color='gray')
ax[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Test ROC (legend: AUC)')
ax[1].set(xlabel='Recall', ylabel='Precision', title='Test precision–recall (legend: AP)')
for a in ax:
    a.set(xlim=(0, 1), ylim=(0, 1.03))
    a.legend(fontsize=8, loc='lower left')
fig.tight_layout()
fig.savefig('analysis/results/performance.pdf')
plt.close(fig)

# Development evidence only: show the already-completed finite searches.
cv = pd.read_csv('analysis/results/cv.csv')
v = pd.read_csv('analysis/results/validation_predictions.csv')
summary = pd.read_csv('analysis/results/summary.csv').set_index('item').value
threshold = float(summary['threshold'])
fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
for model, label in [('geometry_ridge', 'Geometry ridge'), ('full_ridge', 'Full ridge')]:
    a = cv[cv.model.eq(model)].groupby('parameter').AP.agg(['mean', 'min', 'max'])
    ax[0].semilogx(a.index, a['mean'], label=label)
    ax[0].fill_between(a.index, a['min'], a['max'], alpha=.12)
ax[0].set(xlabel='Ridge penalty lambda', ylabel='Grouped fold AP', title='Training: mean and fold range')
ax[0].legend(fontsize=8)
# Distinct score cutoffs suffice to describe all validation decision sets.
cutoffs = np.unique(np.r_[0, v.full_ridge, 1])
values = np.array([f1_score(v.y, v.full_ridge.ge(t)) for t in cutoffs])
# Threshold selection is independently checked by verify.py; this script only displays it.
ax[1].step(cutoffs, values, where='pre', color='#2166ac')
ax[1].axvline(threshold, color='black', linestyle='--', label=f'Selected {threshold:.3f}')
ax[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Decision threshold', ylabel='Validation F1', title='Validation: operating-point selection')
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig('analysis/results/development.pdf')
plt.close(fig)

