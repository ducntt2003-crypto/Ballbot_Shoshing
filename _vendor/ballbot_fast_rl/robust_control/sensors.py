"""Simulated devices -> causal measurement contract; no environment in consumer/estimator."""
from dataclasses import dataclass
import heapq
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT=dict(mode='estimated',gyro_sigma=.001,gyro_bias=[0,0,0],orientation_sigma=.001,position_sigma=.001,velocity_sigma=.003,encoder_sigma=.02,imu_hz=250,attitude_hz=100,position_hz=50,encoder_hz=100,estimator_hz=100,delay_s=0.,dropout=0.,stale_s=.10,process_accel_psd=.04)
@dataclass
class Measurement:
    timestamp:float
    source_timestamp:float
    sequence:int
    p:np.ndarray
    v:np.ndarray
    R:np.ndarray
    omega_B:np.ndarray
    wheels:np.ndarray
    valid:bool
    age:float
    covariance_pv:np.ndarray
    reason:str='OK'
    field_timestamps:dict=None
    prediction_timestamp:float=None
    def control_state(self):return dict(p=self.p.copy(),v=self.v.copy(),R=self.R.copy(),omega_B=self.omega_B.copy(),rpy=Rotation.from_matrix(self.R).as_euler('xyz'))

def validate(m,now,stale=.1):
    finite=np.isfinite(np.r_[m.p,m.v,m.R.ravel(),m.omega_B,m.wheels,m.timestamp,m.source_timestamp,m.age,m.covariance_pv.ravel()]).all()
    if not finite:return False,'NONFINITE'
    if m.timestamp>now+1e-9 or m.source_timestamp>m.timestamp+1e-9 or m.age< -1e-9:return False,'FUTURE_TIMESTAMP'
    if now-m.source_timestamp>stale+1e-9:return False,'STALE'
    if abs(np.linalg.det(m.R)-1)>1e-5 or not np.allclose(m.R.T@m.R,np.eye(3),atol=1e-5):return False,'BAD_FRAME'
    return (True,'OK') if m.valid else (False,m.reason)
class CVFilter:
    """Position-only linear KF at capture timestamps; predicts to now after delayed delivery."""
    def __init__(self,q=.04):self.q=q;self.reset()
    def reset(self):self.x=None;self.P=None;self.stamp=None
    def predict_at(self,t):
        dt=t-self.stamp
        if dt< -1e-9:raise ValueError('Out-of-order capture timestamp')
        A=np.array([[1,dt],[0,1.]])
        Q=self.q*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
        return A@self.x,A@self.P@A.T+Q
    def update(self,stamp,p,sigma):
        p=np.asarray(p,float)
        if self.x is None:self.x=np.vstack([p,np.zeros(3)]);self.P=np.diag([max(sigma**2,1e-10),.1**2]);self.stamp=stamp;return
        x,P=self.predict_at(stamp);variance=max(sigma**2,1e-10);K=P[:,0]/(P[0,0]+variance);self.x=x+K[:,None]*(p-x[0]);I=np.eye(2)-np.outer(K,[1.,0.]);self.P=I@P@I.T+np.outer(K,K)*variance;self.stamp=stamp
    def at(self,now):
        x,P=self.predict_at(now);return x[0],x[1],np.array([P[0,0]]*3+[P[1,1]]*3)
class SimulatedDevices:
    """Only class allowed to see perfect state, solely to generate device proxies."""
    def __init__(self,config,seed):self.cfg=dict(DEFAULT,**config);self.rng=np.random.default_rng(seed);self.next={k:0. for k in ['gyro','attitude','position','encoder']};self.queue=[];self.seq=0
    def sample(self,now,truth,wheels):
        c=self.cfg
        for kind,rate in [('gyro',c['imu_hz']),('attitude',c['attitude_hz']),('position',c['position_hz']),('encoder',c['encoder_hz'])]:
            if now+1e-10<self.next[kind]:continue
            self.next[kind]+=1/rate
            if self.rng.random()<c['dropout']:continue
            if kind=='gyro':value=truth['omega_B']+np.asarray(c['gyro_bias'])+self.rng.normal(0,c['gyro_sigma'],3)
            elif kind=='attitude':value=truth['R']@Rotation.from_rotvec(self.rng.normal(0,c['orientation_sigma'],3)).as_matrix()
            elif kind=='position':value=(truth['p']+self.rng.normal(0,c['position_sigma'],3),truth['v']+self.rng.normal(0,c['velocity_sigma'],3))
            else:value=np.asarray(wheels)+self.rng.normal(0,c['encoder_sigma'],3)
            self.seq+=1;heapq.heappush(self.queue,(now+c['delay_s'],self.seq,kind,now,value))
    def receive(self,now):
        rows=[]
        while self.queue and self.queue[0][0]<=now+1e-10:rows.append(heapq.heappop(self.queue))
        return rows
class MeasurementPipeline:
    def __init__(self,config):self.cfg=dict(DEFAULT,**config);self.reset()
    def reset(self):self.last={};self.kf=CVFilter(self.cfg['process_accel_psd']);self.seq=0;self.last_delivery=-np.inf;self.last_estimate=None;self.last_now=-np.inf
    def ingest(self,packets):
        for delivery,seq,kind,stamp,value in packets:
            if kind in self.last and stamp<=self.last[kind][0]:raise ValueError('Out-of-order source sample')
            flat=np.r_[value[0],value[1]] if kind=='position' else np.asarray(value).ravel()
            if not np.isfinite(flat).all():continue
            self.last[kind]=(stamp,value)
            if kind=='position' and self.cfg['mode']=='estimated':self.kf.update(stamp,value[0],self.cfg['position_sigma'])
    def read(self,now):
        if now<self.last_now-1e-9:raise ValueError('Clock reversed; reset pipeline')
        self.last_now=now;self.seq+=1;c=self.cfg;ready=all(k in self.last for k in ['position','attitude','gyro','encoder'])
        if not ready:return Measurement(now,now,self.seq,np.zeros(3),np.zeros(3),np.eye(3),np.zeros(3),np.zeros(3),False,0.,np.zeros(6),'INITIALIZING')
        stamp=min(x[0] for x in self.last.values());age=now-stamp
        if any(v[0]>now+1e-9 for v in self.last.values()):return Measurement(now,now,self.seq,np.zeros(3),np.zeros(3),np.eye(3),np.zeros(3),np.zeros(3),False,0.,np.zeros(6),'FUTURE_TIMESTAMP')
        # Fusion output update rate is distinct from device acquisition and controller rate.
        if self.last_estimate is None or now-self.last_delivery>=1/c['estimator_hz']-1e-9:
            if c['mode']=='estimated':p,v,cov=self.kf.at(now)
            else:
                p,v=self.last['position'][1];p=p.copy();v=v.copy();cov=np.array([c['position_sigma']**2]*3+[c['velocity_sigma']**2]*3)
            R=self.last['attitude'][1].copy();omega=self.last['gyro'][1].copy();wheels=self.last['encoder'][1].copy()
            self.last_estimate=(p,v,R,omega,wheels,cov,stamp);self.last_delivery=now
        p,v,R,omega,wheels,cov,stamp=self.last_estimate;age=now-stamp;m=Measurement(self.last_delivery,stamp,self.seq,p.copy(),v.copy(),R.copy(),omega.copy(),wheels.copy(),age<=c['stale_s'],age,cov.copy(),'STALE' if age>c['stale_s'] else 'OK');valid,reason=validate(m,now,c['stale_s']);m.valid=valid;m.reason=reason;m.field_timestamps={k:v[0] for k,v in self.last.items()};return m

class PredictionPipeline(MeasurementPipeline):
    """Single controlled variant: causal gyro propagation for fast tilt loop, same slow PV KF."""
    def reset(self):super().reset();self.gyro_history=[]
    def ingest(self,packets):
        super().ingest(packets)
        for _,_,kind,stamp,value in packets:
            if kind=='gyro' and np.isfinite(value).all():self.gyro_history.append((stamp,np.array(value,float)))
        if self.gyro_history:
            minimum=self.gyro_history[-1][0]-.25;self.gyro_history=[p for p in self.gyro_history if p[0]>=minimum]
    def read(self,now):
        m=super().read(now)
        if not m.valid or not self.gyro_history:return m
        attstamp,R=self.last['attitude'];hist=self.gyro_history
        def rate_at(t):
            if t<=hist[0][0]:return hist[0][1]
            for (a,wa),(b,wb) in zip(hist[:-1],hist[1:]):
                if a<=t<=b:return wa+(wb-wa)*((t-a)/(b-a))
            stamp,w=hist[-1]
            if len(hist)<2:return w
            previous,wp=hist[-2];return w+(w-wp)/max(stamp-previous,1e-9)*min(max(t-stamp,0),.01)
        cuts=[attstamp]+[t for t,_ in hist if attstamp<t<now]+[now];R=R.copy()
        for a,b in zip(cuts[:-1],cuts[1:]):
            if b>a:R=R@Rotation.from_rotvec(.5*(rate_at(a)+rate_at(b))*(b-a)).as_matrix()
        m.R=R;m.omega_B=rate_at(now).copy();m.prediction_timestamp=now;return m
