import unittest
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from bulletarm_baselines.fc_dqn.utils.parameters import *

import matplotlib.pyplot as plt

from bulletarm import env_factory

class block_stacking_perfect_classifier(nn.Module):
  def __init__(self):

    super(block_stacking_perfect_classifier, self).__init__()
  
  def check_equal(self, a ,b):
    return abs(a-b)<0.001

  def forward(self,obs,inhand):
    len = obs.shape[0]
    res = []
    for i in range(len):
      obs_height = np.max(obs[i])
      in_hand_height = np.max(inhand[i])
      if (self.check_equal(obs_height,0.03) and self.check_equal(in_hand_height,0)):
        res.append(6)
        continue
      if (self.check_equal(obs_height,0.03) and self.check_equal(in_hand_height,0.03)):
        res.append(5)
        continue
      if (self.check_equal(obs_height,0.06) and self.check_equal(in_hand_height,0)):
        res.append(4)
        continue
      if (self.check_equal(obs_height,0.06) and self.check_equal(in_hand_height,0.03)):
        res.append(3)
        continue
      if (self.check_equal(obs_height,0.09) and self.check_equal(in_hand_height,0)):
        res.append(2)
        continue
      if (self.check_equal(obs_height,0.09) and self.check_equal(in_hand_height,0.03)):
        res.append(1)
        continue
      if (self.check_equal(obs_height,0.12) and self.check_equal(in_hand_height,0)):
        res.append(0)
        continue
      raise NotImplementedError(f'error classifier with obs_height = {obs_height}, in_hand_height = {in_hand_height}')
    
    return torch.tensor(res).to(device)



class TestBulletBlockStacking(unittest.TestCase):
  env_config = {'num_objects': 4}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(5, 'block_stacking', self.env_config, self.planner_config)
    classifer = block_stacking_perfect_classifier()
    (states_, in_hands_, obs_) = env.reset()
    print(states_,in_hands_.shape,obs_.shape)
    abs_state = (4)*2 - 2 
    for i in range(5, -1, -1):
      action = env.getNextAction()
      print(classifer(obs_,in_hands_),abs_state)
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      abs_state -= 1
    print(classifer(obs_,in_hands_),abs_state)
    print(f'success at state {abs_state}')
    env.close()

  def testPlanner2(self):
    self.env_config['render'] = False
    self.env_config['seed'] = 0
    num_processes = 20
    env = env_factory.createEnvs(num_processes,  'block_stacking', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    env.reset()
    pbar = tqdm(total=500)
    while total < 500:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=True)
      s += rewards.sum()
      total += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
      pbar.update(dones.sum())
    env.close()

if __name__ == '__main__':
  env = TestBulletBlockStacking()
  env.testPlanner()