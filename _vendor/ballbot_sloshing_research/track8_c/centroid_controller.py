"""D3 revision: feedback of identified coupled modal coordinates, same backbone."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.linalg import eig
from scipy.signal import place_poles
from ballbot_fast_rl.outer_loop.control import MIX,heading
from ballbot_sloshing_research.fluid.reduced_order import ModalParameters
from .common import *
from .centroid_id import coordinates,eigensummary
class CentroidModalFeedback:
 name='TRACK8C_CENTROID_MODAL_FEEDBACK_V4'
 def __init__(self):
  self.cfg=read(ROOT/'centroid_controller_config.json');self.K=np.array(self.cfg['K']);self.scale=np.array(self.cfg['scale']);self.keep=self.cfg['keep'];self.wheel_map=np.array(self.cfg['nominal_wheel_map']);self.calls=0;self.clipped=0
 def action(self,e):
  p,v,a=e.target();yaw=Rotation.from_matrix(e.data.xmat[e.base].reshape(3,3)).as_euler('xyz')[2];rp=np.linalg.solve(np.array(e.backbone.cfg['G']),heading(yaw).T@a);R=Rotation.from_euler('xyz',[*rp,yaw]);qref=np.array(self.cfg['anchor']);qref[e.base_q:e.base_q+2]=p;qref[e.base_q+3:e.base_q+7]=R.as_quat()[[3,0,1,2]];qref[e.ball_q:e.ball_q+2]=p+(R.as_matrix()@np.array(self.cfg['base_to_ball_B']))[:2];qref[e.slosh_q]=(R.as_matrix().T@e.model.opt.gravity)[:2]/e.pfluid.omega**2-(heading(yaw).T@a)/e.pfluid.omega**2;qref[e.wheel_qpos]=self.wheel_map@np.r_[heading(yaw).T@(p-e.origin[:2]),rp,0.];ball=e.model.body('ball').id;Rb=e.data.xmat[ball].reshape(3,3);qref[e.ball_q+3:e.ball_q+7]=e.data.qpos[e.ball_q+3:e.ball_q+7];center=np.r_[p+(R.as_matrix()@np.array(self.cfg['base_to_ball_B']))[:2],self.cfg['anchor_ball_center_z']];qref[e.ball_q:e.ball_q+3]=center-Rb@e.model.geom_pos[e.ball_geom]
  state=coordinates(e,qref,self.keep);nominal=np.zeros_like(state);n=len(self.keep);vb=heading(yaw).T@v;velocity=np.zeros(e.model.nv);velocity[e.base_d:e.base_d+2]=v;velocity[e.ball_d:e.ball_d+2]=v;velocity[e.wheel_dofs]=self.wheel_map@np.r_[vb,[0,0,0]];ball=e.model.body('ball').id;world_omega=e.data.xmat[ball].reshape(3,3)@e.data.qvel[e.ball_d+3:e.ball_d+6];radius=e.model.geom_size[e.ball_geom,0];velocity[e.ball_d+3:e.ball_d+6]=[-v[1]/radius,v[0]/radius,world_omega[2]];velocity[e.base_d+5]=e.data.qvel[e.base_d+5];nominal[n:n+e.model.nv]=velocity;nominal[n+e.model.nv+4:n+e.model.nv+6]=rp
  virtual=self.K@((state-nominal)/self.scale);delta=MIX@virtual;factor=min(1,.3/max(1e-15,float(abs(delta).max())));base=-10*e.base_action
  for b,d in zip(base,delta):
   if d>1e-15:factor=min(factor,(10-b)/d)
   elif d<-1e-15:factor=min(factor,(-10-b)/d)
  factor=max(0,factor);self.calls+=1;self.clipped+=factor<1-1e-12
  return np.clip(-factor*delta/.3,-1,1)
def main():
 # Mean of all9correctedinvariantJacobians, no beststate/pole/gainselection.
 path=ROOT/f'centroid_model_{REPRO[0]}_5.npz';z=np.load(path);models=read(ROOT/'centroid_identification.json')['records'];A=np.mean([np.load(r['path'])['A'] for r in models],axis=0);B=np.mean([np.load(r['path'])['B'] for r in models],axis=0);keep=z['keep'].tolist();scale=z['scale'];val,L,V=eig(A,left=True,right=True);selected=[i for i,v in enumerate(val) if 1<np.angle(v)/(.01*2*np.pi)<4 and np.log(abs(v))/.01>0 and np.linalg.norm(V[[keep.index(int(j)) for j in z['slosh_dofs']],i])**2/np.linalg.norm(V[:,i])**2>.1];assert len(selected)==1,selected
 W=[];blocks=[];poles=[];p=ModalParameters();decay=p.zeta*p.omega
 for i in selected:
  row=L[:,i].conj();row/=row@V[:,i];W.extend([row.real,row.imag]);a,b=val[i].real,val[i].imag;blocks.append(np.array([[a,-b],[b,a]]));theta=np.angle(val[i]);target=np.exp(-decay*.01+1j*theta);poles.extend([target,target.conjugate()])
 from scipy.linalg import block_diag
 W=np.array(W);Az=block_diag(*blocks);Bz=W@B;placement=place_poles(Az,Bz,poles,method='YT',rtol=1e-7,maxiter=100);K=-placement.gain_matrix@W;diagnostics=[]
 for r in read(ROOT/'centroid_identification.json')['records']:
  zz=np.load(r['path']);diagnostics.append(dict(seed=r['seed'],time_s=r['time_s'],predicted_modes=eigensummary(zz['A']+zz['B']@K,keep,len(scale)-len(keep)-9,z['slosh_dofs'])))
 from .native import restore_snapshot
 e=restore_snapshot(ROOT/'reproduction'/str(REPRO[0])/'snapshot_5.pkl')
 try:offset=e.data.xmat[e.base].reshape(3,3).T@(e.data.geom_xpos[e.ball_geom]-e.data.xpos[e.base])
 finally:e.close()
 X=[];Y=[]
 for r in read(OLD/'all_development_results.json'):
  if r['controller']=='V2_FFv1' and r['payload']=='rigid' and r['kind']=='figure8' and r['A']==.5:
   zz=np.load(r['trace_path']);x=zz['rows'];vel=zz['qvel'];m=x[:,0]>3
   X.append(np.array([np.r_[heading(a[13]).T@a[3:5],b[3:6]] for a,b in zip(x[m],vel[m])]));Y.append(x[m,22:25])
 wheelmap=np.linalg.lstsq(np.concatenate(X),np.concatenate(Y),rcond=None)[0].T
 save('centroid_controller_revision_addendum.json',dict(reason='V1localinstantaneousworkmapdidnotstabilizeclosedloop. NativeaugmentedJacobianhasgrowing2Hzpairs; V1predictedgrowthpersists/increases. ReviseoneD3strategyusingactualcontrollerstates/delay, notchangingphysicaldamping orblindgain sweep. V1retainedfailed; no othercontrollerfamilytrial.',design='Meanall9invariantmodels; retainwheelanglesandworldballomega. SelectonlyONEgrowingfluid-dominantpair1-4Hz (growth>0). Stablemodesnotretuned. Correctballposition/linearvelocitytoGEOMCENTER, worldballomega; bodyoriginis14cmoffCOM. All40statesconsistent. PlaceONLYthese4realmodalcoordinatesatnaturalpassiveamplitudedecayzeta*omega, preservefrequencies. Otherstatesunmodified. No optimizationovervalidation/finaldata.',native_new_budget='Reuse remaining600sdevelopmentallocation (V1elapsed accounted), same thresholds. Shortstagegates beforeE4. RevisionconfigurationfrozenbeforeanyV2simulation.'))
 save('centroid_controller_config.json',dict(name=CentroidModalFeedback.name,K=K.tolist(),keep=keep,scale=scale.tolist(),anchor=z['anchor'].tolist(),base_to_ball_B=offset.tolist(),nominal_wheel_map=wheelmap.tolist(),anchor_ball_center_z=float(z['anchor'][0]*0+read(ROOT/'centroid_reference_point.json')['ball_center_z']),modal_projection=W.tolist(),placed_modal_poles=[dict(real=float(v.real),imag=float(v.imag)) for v in poles],desired_decay_per_s=decay,local_model_diagnostics=diagnostics,privileged=True,residual_bound_Nm=.3,backbone='OriginalV2_FFv1unchanged',reference='OriginalTrack8p/v/a unchanged; nominalstateonlyforfeedbackerrors, neverreplacesreferenceorphysicalstate',scope='Localfrozen-timepoleplacement, notglobalnonlinearstabilityproof; initializedzeroI/nominalvel/tilt, fullphysical/controllerstatefeedbackprivileged. Nativevalidationrequired.'))
 print('CENTROID_CONTROLLER_V4',K.shape,[(val[i],np.log(abs(val[i]))/.01) for i in selected],flush=True)
if __name__=='__main__':main()
