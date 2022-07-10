import unittest
import time
import numpy as np

import matplotlib.pyplot as plt

from bulletarm import env_factory
from bulletarm.envs.utils.check_goal import CheckGoal

class TestBulletHouseXDeconstruct(unittest.TestCase):
  env_config = {}
  planner_config = {'random_orientation': True}

  def testPlanner(self, goal_str, num_episodes=10):
    print(goal_str)
    self.env_config['render'] = True
    self.env_config['goal_string'] = goal_str
    env = env_factory.createEnvs(1, 'house_building_x_deconstruct', self.env_config, self.planner_config)
    num_ep = 0

    while (num_ep < num_episodes):
      env.reset()
      time.sleep(12)
      dones = [0]
      i = 0
      self.check_goal = CheckGoal(self.env_config['goal_string'], env)
      self.check_goal.parse_goal_()
      while (dones[0] != 1.0):
        action = env.getNextAction()
        (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
        i += 1
        # print(dones, i)
      self.assertEqual(2*self.check_goal.num_objects - 2, i)
      num_ep += 1
    env.close()

cls = TestBulletHouseXDeconstruct()
for goal_str in ['1l1l2r']:
  cls.testPlanner(goal_str)
# for goal_str in ['1b1r', '2b1r', '2b2r', '1l1r', '1l2r', '1b1b1r', '2b1b1r', '2b2b1r', '2b2b2r', '2b1l1r', '2b1l2r', '1l1l1r', '1l1l2r', '1l2b2r', '1l2b1r']:
  # cls.testPlanner(goal_str)
