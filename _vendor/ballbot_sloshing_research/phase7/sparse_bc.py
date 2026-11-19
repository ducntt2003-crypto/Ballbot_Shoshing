"""One two-head sparse architecture, train-only class weight, immutable checkpoints."""
import time,numpy as np,torch
from torch import nn
from .common import *
from .metrics import sparse_metrics
class Net(nn.Module):
 def __init__(self,dim=23):
  super().__init__();self.trunk=nn.Sequential(nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh());self.gate=nn.Linear(128,1);self.action=nn.Linear(128,3)
 def forward(self,x):
  h=self.trunk(x);return self.gate(h).squeeze(-1),.5*torch.tanh(self.action(h))
class Student:
 def __init__(self,path,threshold=None):
  meta=read(Path(path).with_suffix('.json'));self.mode=meta['condition'];self.net=Net(23 if self.mode=='O1' else 27);self.net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));self.net.eval();execution=read(Path(path).parent/'validated_execution.json') if (Path(path).parent/'validated_execution.json').exists() else {};self.threshold=threshold if threshold is not None else execution.get('threshold',meta.get('threshold',.7));self.last_probability=0.;self.last_magnitude=np.zeros(3)
 def parts(self,o):
  with torch.no_grad():logit,a=self.net(torch.as_tensor(np.asarray(o)[None],dtype=torch.float32));p=float(logit.sigmoid().item());v=a.numpy()[0]
  self.last_probability=p;self.last_magnitude=v.copy();return p,v
 def predict(self,o,index=0):
  p,a=self.parts(o);return a if p>=self.threshold else np.zeros(3)
 def batch(self,x,threshold=None):
  with torch.no_grad():l,a=self.net(torch.as_tensor(x,dtype=torch.float32));p=l.sigmoid().numpy();a=a.numpy()
  return p,a,a*(p>= (self.threshold if threshold is None else threshold))[:,None]
def load(split,mode='O1'):
 x=[];y=[]
 for r in read(ROOT/'event_dataset_registry.json')['records']:
  if r['split']!=split:continue
  z=np.load(r['path']);m=z['valid'];x.append(z[mode][m]);y.append(z['action'][m])
 return np.concatenate(x).astype(np.float32),np.concatenate(y).astype(np.float32)
def fit(seed):
 torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed);x,y=load('train');v,t=load('validation');active=np.linalg.norm(y,axis=1)>.05;weight=float((~active).sum()/active.sum());model=Net();opt=torch.optim.Adam(model.parameters(),lr=.001);X=torch.as_tensor(x);Y=torch.as_tensor(y);G=torch.as_tensor(active,dtype=torch.float32);V=torch.as_tensor(v);T=torch.as_tensor(t);validactive=torch.linalg.norm(T,dim=1)>.05;logs=[];best=float('inf');beststate=None;start=time.monotonic();pos=torch.tensor(weight)
 for update in range(1000):
  idx=rng.integers(0,len(x),256);l,a=model(X[idx]);mask=G[idx]>0;gate=nn.functional.binary_cross_entropy_with_logits(l,G[idx],pos_weight=pos);mag=((a[mask]-.0-Y[idx][mask])/.5).square().mean() if mask.any() else a.sum()*0;loss=gate+mag;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
  if (update+1)%100==0:
   with torch.no_grad():vl,va=model(V);vg=nn.functional.binary_cross_entropy_with_logits(vl,validactive.float(),pos_weight=pos);vm=((va[validactive]-T[validactive])/.5).square().mean();score=float(vg+vm)
   logs.append(dict(update=update+1,train_gate_loss=float(gate.detach()),train_active_loss=float(mag.detach()),validation_loss=score))
   if score<best:best=score;beststate={k:a.detach().clone() for k,a in model.state_dict().items()}
  if time.monotonic()-start>120:break
 model.load_state_dict(beststate);folder=ROOT/'training'/f'sparse_seed_{seed}';folder.mkdir(parents=True,exist_ok=False);path=folder/'student.pt';torch.save(model.state_dict(),path)
 meta=dict(experiment=ID,condition='O1',seed=seed,threshold=.7,architecture='shared128x128Tanh, gate1, action3*.5tanh',training_samples=len(x),validation_samples=len(v),class_positive_weight=weight,updates=update+1,wall_s=time.monotonic()-start,checkpoint_sha256=sha(path),logs=logs)
 with path.with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2,allow_nan=False)
 return meta|dict(checkpoint=str(path))
def main():
 assert read(ROOT/'teacher_replay.json')['status']=='TEACHER_REPLAY_VALIDATED';rows=[fit(s) for s in TRAIN];save('sparse_training.json',dict(experiment=ID,runs=rows,source_hashes=source_hashes()));print('SPARSE_TRAINED',len(rows),flush=True)
if __name__=='__main__':main()
