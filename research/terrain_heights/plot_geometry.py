from pathlib import Path
import json,sys,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));from experiment import terrain
P=json.loads((R/'protocol.json').read_text());G=json.loads((R/'geometry.json').read_text());axis=np.linspace(-2,2,129)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for ax,c in zip(axes.ravel(),P['conditions']):
 H=terrain(c);im=ax.imshow(H*1000,extent=[-2,2,-2,2],origin='lower',cmap='terrain',vmin=0,vmax=20);t=np.linspace(0,30,301);ax.plot(.75*np.sin(2*np.pi*t/30),.375*np.sin(4*np.pi*t/30),'k--',lw=1,label='Desired figure8');ax.set(xlim=(-1.5,1.5),ylim=(-1.1,1.1),xlabel='X (m)',ylabel='Y (m)',title=f'{c}: peak {G[c]["max_height_m"]*1000:.2f} mm, max slope {G[c]["max_slope_deg"]:.2f} deg');ax.legend(fontsize=8);fig.colorbar(im,ax=ax,label='Collision height (mm)')
fig.suptitle('New collision terrains: different real heights; no changes to robot, controller or torque limits');fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(R/'figures'/('01_height_maps.'+ext),dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4))
from ballbot_sloshing_research.track8.terrain import sample
for c in P['conditions']:
 H=terrain(c);t=np.linspace(0,30,501);ref=np.c_[.75*np.sin(2*np.pi*t/30),.375*np.sin(4*np.pi*t/30)];z=[1000*sample(H,*p)[0] for p in ref];ax.plot(t,z,label=c)
ax.set(xlabel='Reference time (s)',ylabel='Height under desired path (mm)',title='Terrain elevation along the same figure-eight; these are geometry profiles, not robot results');ax.grid(alpha=.3);ax.legend();fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(R/'figures'/('02_reference_height_profile.'+ext),dpi=180)
plt.close(fig);print('GEOMETRY_PLOTS_COMPLETE')
