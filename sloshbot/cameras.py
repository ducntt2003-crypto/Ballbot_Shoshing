"""Camera-only transforms; no model/data writes."""
import numpy as np

def configure_camera(cam,model,data,mode=2):
    base=model.body('base').id;p=data.xpos[base];R=data.xmat[base].reshape(3,3);yaw=np.arctan2(R[1,0],R[0,0])
    cam.lookat[:]=p if mode==1 else [.32,0,.30];cam.distance=1.2 if mode==1 else 1.8;cam.azimuth=float(np.degrees(yaw)+180) if mode==1 else 125;cam.elevation=-89 if mode==3 else -8 if mode==4 else -25
    if mode==5:cam.lookat[:]=data.xpos[model.body('water_tank').id]+[0,0,.035];cam.distance=.29;cam.azimuth=110;cam.elevation=-45
    return cam
