import json
from pathlib import Path
from .env import BallBotFastBalanceEnv,OBS_CONTRACT,ACTION_CONTRACT,REWARD_VERSION

def metadata(config):
 e=BallBotFastBalanceEnv(config['env'])
 result=dict(observation_contract=OBS_CONTRACT,action_contract=ACTION_CONTRACT,reward_version=REWARD_VERSION,environment_version='fast_env_1',robot_model_hash=e.model_hash,normalization_config=dict(running=False,embedded_static_scaling=True),observation_dim=18,action_dim=3,physics_dt=e.physics_dt,control_dt=e.dt,reward_config=e.cfg['reward_weights'],environment_config=e.cfg)
 e.close();return result

def verify(checkpoint,config,allow_model_transfer=False):
 p=Path(str(checkpoint).removesuffix('.zip')+'.metadata.json');saved=json.loads(p.read_text());expected=metadata(config)
 for key in ['observation_contract','action_contract','reward_version','robot_model_hash','normalization_config','control_dt','reward_config']:
  if key=='robot_model_hash' and allow_model_transfer:continue
  if saved[key]!=expected[key]:raise ValueError('Incompatible checkpoint contract: '+key)
 return saved
