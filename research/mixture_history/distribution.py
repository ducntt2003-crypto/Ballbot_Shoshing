"""TRACK8Y_STRUCTURED_RESIDUAL_EXPLORATION_V1: stateless marginalized axis mixture."""
import math,torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

class AxisMixture(DiagGaussianDistribution):
 def __init__(self,action_dim=3):
  super().__init__(action_dim);self.offsets=torch.tensor([[0,0,0],[.5,0,0],[-.5,0,0],[0,.5,0],[0,-.5,0],[0,0,.5],[0,0,-.5]],dtype=torch.float32);self.weights=torch.tensor([.7,.05,.05,.05,.05,.05,.05]);g=torch.Generator().manual_seed(2277771);eps=torch.randn((16,3),generator=g);self.eps=torch.cat([eps,-eps])
 def proba_distribution(self,mean_actions,log_std):
  self.mean_actions=mean_actions;self.std=log_std.exp().expand_as(mean_actions);self.centers=mean_actions[:,None,:]+self.offsets.to(mean_actions)[None];self.w=self.weights.to(mean_actions);comp=torch.distributions.Independent(torch.distributions.Normal(self.centers,self.std[:,None,:].expand_as(self.centers)),1);mix=torch.distributions.Categorical(probs=self.w.expand(mean_actions.shape[0],-1));self.distribution=torch.distributions.MixtureSameFamily(mix,comp);return self
 def log_prob(self,actions):return self.distribution.log_prob(actions)
 def sample(self):return self.distribution.sample()
 def mode(self):return self.mean_actions
 def entropy(self):
  points=self.centers[:,:,None,:]+self.std[:,None,None,:]*self.eps.to(self.mean_actions)[None,None,:,:];diff=(points[:,:,:,None,:]-self.centers[:,None,None,:,:])/self.std[:,None,None,None,:];lp=(-.5*diff.square()-torch.log(self.std[:,None,None,None,:])-.5*math.log(2*math.pi)).sum(-1)+torch.log(self.w)[None,None,None,:];return -(torch.logsumexp(lp,dim=-1).mean(-1)*self.w[None,:]).sum(-1)
class MatchedPolicy(ActorCriticPolicy):
 def __init__(self,*args,structured=False,**kwargs):
  super().__init__(*args,**kwargs);self.structured=structured
  if structured:self.action_dist=AxisMixture(self.action_space.shape[0])
 def _get_constructor_parameters(self):
  d=super()._get_constructor_parameters();d['structured']=self.structured;return d
