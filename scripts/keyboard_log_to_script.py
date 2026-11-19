"""Convert one recorded GUI episode to high-level key requests, never motor torques."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--episode',type=int,required=True);p.add_argument('--duration',type=float,required=True);p.add_argument('--output',required=True);a=p.parse_args()
events=[dict(t=0.,keys=[])];previous=[]
for line in Path(a.input).read_text().splitlines():
 r=json.loads(line)
 if r['episode']!=a.episode:continue
 t=float(r['sim_t']);keys=r['keys']
 if t>a.duration:continue
 if keys!=previous:events.append(dict(t=t,keys=keys));previous=keys
assert a.duration>0
with Path(a.output).open('x') as f:json.dump(dict(name='recorded_gui_high_level_commands',source_episode=a.episode,duration_s=a.duration,events=events),f,indent=2)
