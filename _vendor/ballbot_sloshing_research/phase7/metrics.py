"""Sparse classification, event timing and amplitude metrics with defined coverage."""
import numpy as np

def bursts(mask,dt):
 m=np.r_[False,np.asarray(mask,bool),False];start=np.flatnonzero(np.diff(m.astype(int))==1);end=np.flatnonzero(np.diff(m.astype(int))==-1)
 return [(float(a*dt),float((b-a)*dt)) for a,b in zip(start,end)]
def sparse_metrics(y,p,dt=.05):
 y=np.asarray(y);p=np.asarray(p);truth=np.linalg.norm(y,axis=1)>.05;pred=np.linalg.norm(p,axis=1)>.05;tp=int(np.sum(truth&pred));fp=int(np.sum(~truth&pred));fn=int(np.sum(truth&~pred));tn=int(np.sum(~truth&~pred));precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.;eligible=truth&pred;cos=(p*y).sum(axis=1)/np.maximum(np.linalg.norm(p,axis=1)*np.linalg.norm(y,axis=1),1e-12)
 a=bursts(truth,dt);b=bursts(pred,dt);pairs=[];useda=set();usedb=set()
 for gap,i,j in sorted((abs(v[0]-w[0]),i,j) for i,v in enumerate(a) for j,w in enumerate(b) if abs(v[0]-w[0])<=.10000001):
  if i not in useda and j not in usedb:useda.add(i);usedb.add(j);pairs.append((i,j))
 return dict(samples=len(y),teacher_zero_fraction=float((~truth).mean()),predicted_active_fraction=float(pred.mean()),TP=tp,FP=fp,FN=fn,TN=tn,precision=precision,recall=recall,F1=2*precision*recall/(precision+recall) if precision+recall else 0.,balanced_accuracy=.5*(recall+tn/(tn+fp)) if truth.any() and (~truth).any() else None,false_positive_rate=fp/(fp+tn) if fp+tn else None,false_negative_rate=fn/(fn+tp) if fn+tp else None,action_RMSE=float(np.sqrt(np.mean((p-y)**2))),nonzero_action_MAE=float(np.mean(abs(p[truth]-y[truth]))) if truth.any() else None,nonzero_direction_accuracy=float(np.mean(cos[eligible]>0)) if eligible.any() else None,direction_coverage=float(eligible.sum()/truth.sum()) if truth.any() else None,true_events=len(a),predicted_events=len(b),matched_events=len(pairs),event_timing_MAE_s=float(np.mean([abs(a[i][0]-b[j][0]) for i,j in pairs])) if pairs else None,event_duration_MAE_s=float(np.mean([abs(a[i][1]-b[j][1]) for i,j in pairs])) if pairs else None,unmatched_true_events=len(a)-len(pairs),unmatched_predicted_events=len(b)-len(pairs))
