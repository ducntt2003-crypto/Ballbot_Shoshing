from pathlib import Path
import sys,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from sloshbot.bootstrap import setup
setup()
from ballbot_sloshing_research.fluid.reduced_order import ModalParameters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=ROOT/'research/high_fidelity';records=[];p=ModalParameters();gates=json.loads((r/'protocol.json').read_text())['screen_gates']
for name in ['dx8_dt50','dx6_dt50','dx6_dt25']:
 x=np.genfromtxt(r/name/'diagnostics.csv',delimiter=',',names=True);meta=json.loads((r/name/'run_meta.json').read_text());status=json.loads((r/name/'run_status.json').read_text());cfg=json.loads((r/(name+'.json')).read_text());valid=x['force_valid']>0;mass=meta['actual_mass_kg'];weight=9.81*mass;com=np.c_[x['com_Tx_m'],x['com_Ty_m'],x['com_Tz_m']];comdrift=float(np.linalg.norm(com-com[0],axis=1).max());force=np.c_[x['Fx_W_N'],x['Fy_W_N'],x['Fz_W_N']];fdef=float(np.linalg.norm(force[valid]-[0,0,-weight],axis=1).max()/weight);denmax=float(max(abs(x['density_min_kg_m3']/998-1).max(),abs(x['density_max_kg_m3']/998-1).max()));volume=[]
 for frame in sorted((r/name/'particles').glob('*.csv')):
  a=np.genfromtxt(frame,delimiter=',',names=True);volume.append(float((meta['particle_mass_kg']/a['density_kg_m3']).sum()))
 row=dict(case=name,spacing_m=cfg['spacing_m'],dt_s=cfg['timestep_s'],particles=meta['fluid_particles'],native_steps=status['completed_steps'],duration_s=status['simulated_seconds'],wall_s=status['wall_seconds'],actual_mass_kg=mass,nominal_ROM_mass_kg=p.mass,relative_mass_mismatch=mass/p.mass-1,pressure_RMSE_peak_Pa=float(x['pressure_rmse_Pa'].max()),RMS_velocity_peak_m_s=float(x['rms_relative_velocity_m_s'].max()),max_velocity_m_s=float(x['max_relative_velocity_m_s'].max()),density_error_max=denmax,COM_drift_max_m=comdrift,wall_weight_defect_relative_peak=fdef,kinetic_energy_peak_J=float(x['K_relative_J'].max()),mass_relative_drift=float(np.ptp(x['mass_kg'])/mass),escaped_max=int(x['escaped'].max()),volume_est_initial_m3=volume[0],volume_est_relative_drift_max=float(max(abs(np.array(volume)/volume[0]-1))),momentum_defect_peak_N=float(x['momentum_balance_defect_N'].max()))
 checks=dict(complete=status['status']=='complete',pressure=row['pressure_RMSE_peak_Pa']<=31.8,velocity=row['RMS_velocity_peak_m_s']<=.005,density=denmax<=.01,COM=comdrift<=.1*cfg['spacing_m'],wall_force=fdef<=.05,mass=row['mass_relative_drift']<=1e-12,containment=row['escaped_max']==0);row['screen_pass']=all(checks.values());row['checks']=checks;records.append(row)
with (r/'screen_summary.json').open('x') as f:json.dump(dict(records=records,status='HIGH_FIDELITY_REFERENCE_IMPLEMENTED',physical_liquid_status='SPH_BLOCKED',native_steps=sum(v['native_steps'] for v in records),dynamic_cases='NOT_RUN_STATIC_GATE_AND_MASS_MATCH_INCOMPLETE',resolution='TWO_SPACING_SCREEN_WITH_MASS_MISMATCH_NOT_SPATIAL_CONVERGENCE',temporal_pair_scope='20ms only'),f,indent=2)
with (r/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=[k for k in records[0] if k!='checks']);w.writeheader();w.writerows([{k:v for k,v in a.items() if k!='checks'} for a in records])
fig,axes=plt.subplots(3,1,figsize=(7,8))
for name in ['dx8_dt50','dx6_dt50','dx6_dt25']:
 x=np.genfromtxt(r/name/'diagnostics.csv',delimiter=',',names=True)
 for ax,key,unit in zip(axes,['pressure_rmse_Pa','rms_relative_velocity_m_s','Fz_W_N'],['Pressure RMSE (Pa)','Relative RMS velocity (m/s)','Fluid-on-tank Fz (N)']):ax.plot(x['time_s'],x[key],label=name);ax.set(xlabel='Time (s)',ylabel=unit);ax.grid(alpha=.3);ax.legend(fontsize=7)
fig.suptitle('Independent Chrono WCSPH static screening; NOT physical validation');fig.tight_layout();fig.savefig(r/'static_screen.png',dpi=180);fig.savefig(r/'static_screen.pdf');plt.close(fig)
print('HIGH_FIDELITY_SCREEN',[(a['case'],a['screen_pass'],a['checks']) for a in records],flush=True)
