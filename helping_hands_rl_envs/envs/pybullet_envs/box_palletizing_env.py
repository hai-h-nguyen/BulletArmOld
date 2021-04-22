import pybullet as pb
import numpy as np

from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.simulators.pybullet.objects.box import Box
from helping_hands_rl_envs.simulators.pybullet.objects.pallet import Pallet
from helping_hands_rl_envs.planners.box_palletizing_planner import BoxPalletizingPlanner

class BoxPalletizingEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.pallet_height = 0.0435
    self.pallet_pos = [0.5, 0.1, self.pallet_height/2]
    self.pallet_size = [0.066*3, 0.1*2, 0.05]
    self.pallet_rz = 0
    self.pallet = None
    self.box_height = 0.056 * self.block_scale_range[1]

    self.pick_offset = 0.02
    self.place_offset = 0.06

    self.object_init_space = np.asarray([[0.3, 0.7],
                                         [-0.3, 0.1],
                                         [0, 0.40]])

    # pos candidate for odd layer
    dx = self.pallet_size[0] / 6
    dy = self.pallet_size[1] / 4
    pos_candidates = np.array([[2 * dx, -dy], [2 * dx, dy],
                               [0, -dy], [0, dy],
                               [-2 * dx, -dy], [-2 * dx, dy]])

    R = np.array([[np.cos(self.pallet_rz), -np.sin(self.pallet_rz)],
                  [np.sin(self.pallet_rz), np.cos(self.pallet_rz)]])
    transformed_pos_candidate = R.dot(pos_candidates.T).T
    self.odd_place_pos_candidate = transformed_pos_candidate + self.pallet_pos[:2]

    # pos candidate for even layer
    dx = self.pallet_size[0] / 4
    dy = self.pallet_size[1] / 6
    pos_candidates = np.array([[dx, -2*dy], [dx, 0], [dx, 2*dy],
                               [-dx, -2*dy], [-dx, 0], [-dx, 2*dy]])

    R = np.array([[np.cos(self.pallet_rz), -np.sin(self.pallet_rz)],
                  [np.sin(self.pallet_rz), np.cos(self.pallet_rz)]])
    transformed_pos_candidate = R.dot(pos_candidates.T).T
    self.even_place_pos_candidate = transformed_pos_candidate + self.pallet_pos[:2]

  def getValidSpace(self):
    return self.object_init_space

  def _getExistingXYPositions(self):
    positions = [o.getXYPosition() for o in self.objects]
    for pos in self.odd_place_pos_candidate:
      positions.append(list(pos))
    return positions

  def generateOneBox(self):
    while True:
      try:
        self._generateShapes(constants.BOX, 1, random_orientation=self.random_orientation)
      except NoValidPositionException:
        continue
      else:
        break

  def reset(self):
    if self.pallet is not None:
      pb.removeBody(self.pallet.object_id)
    self.resetPybulletEnv()
    self.pallet = Pallet(self.pallet_pos, transformations.quaternion_from_euler(0, 0, self.pallet_rz), 1)
    self.generateOneBox()
    return self._getObservation()

  def step(self, action):
    self.takeAction(action)
    self.wait(100)
    if self.isSimValid() and len(self.objects) < self.num_obj and not self._isHolding():
      self.generateOneBox()
    obs = self._getObservation(action)
    done = self._checkTermination()
    reward = 1.0 if done else 0.0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def isSimValid(self):
    # n_obj_on_ground = len(list(filter(lambda o: self._isObjOnGround(o), self.objects)))
    # level1_threshold = self.pallet_height + 0.5*self.box_height - 0.01
    # level2_threshold = self.pallet_height + 1.5*self.box_height - 0.01
    # level3_threshold = self.pallet_height + 2.5*self.box_height - 0.01
    # level4_threshold = self.pallet_height + 3.5*self.box_height - 0.01
    # level1_objs = list(filter(lambda o: level1_threshold<o.getZPosition()<level2_threshold, self.objects))
    # level2_objs = list(filter(lambda o: level2_threshold<o.getZPosition()<level3_threshold, self.objects))
    # level3_objs = list(filter(lambda o: level3_threshold<o.getZPosition()<level4_threshold, self.objects))
    # n_level1 = len(level1_objs)
    # n_level2 = len(level2_objs)
    # n_level3 = len(level3_objs)
    # level1_rz = list(map(lambda o: transformations.euler_from_quaternion(o.getRotation())[2], level1_objs))
    # level2_rz = list(map(lambda o: transformations.euler_from_quaternion(o.getRotation())[2], level2_objs))
    # level3_rz = list(map(lambda o: transformations.euler_from_quaternion(o.getRotation())[2], level3_objs))
    # level1_rz_goal = self.pallet_rz + np.pi/2
    # if level1_rz_goal > np.pi:
    #   level1_rz_goal -= np.pi
    # level2_rz_goal = self.pallet_rz
    # if level2_rz_goal > np.pi:
    #   level2_rz_goal -= np.pi
    # level3_rz_goal = level1_rz_goal
    # def rz_close(rz, goal):
    #   angle_diff = abs(rz - goal)
    #   angle_diff = min(angle_diff, abs(angle_diff - np.pi))
    #   return angle_diff < np.deg2rad(20)
    # level1_rz_ok = all(map(lambda rz: rz_close(rz, level1_rz_goal), level1_rz))
    # level2_rz_ok = all(map(lambda rz: rz_close(rz, level2_rz_goal), level2_rz))
    # level3_rz_ok = all(map(lambda rz: rz_close(rz, level3_rz_goal), level3_rz))

    n_obj_on_ground = len(list(filter(lambda o: self._isObjOnGround(o), self.objects)))
    n_level1, n_level2, n_level3 = self.getNEachLevel()
    all_upright = all(map(lambda o: self._checkObjUpright(o), self.objects))
    rz_valid = self.checkRzValid()

    return n_obj_on_ground <= 1 and n_level1 <= 6 and n_level2 <= 6 and n_level3 <= 6 and rz_valid and all_upright

  def _checkTermination(self):
    n_level1, n_level2, n_level3 = self.getNEachLevel()
    all_upright = all(map(lambda o: self._checkObjUpright(o), self.objects))
    rz_valid = self.checkRzValid()
    return n_level1 == 6 and n_level2 == 6 and n_level3 == 6 and rz_valid and all_upright


  def getObjEachLevel(self):
    level1_threshold = self.pallet_height + 0.5 * self.box_height - 0.01
    level2_threshold = self.pallet_height + 1.5 * self.box_height - 0.01
    level3_threshold = self.pallet_height + 2.5 * self.box_height - 0.01
    level4_threshold = self.pallet_height + 3.5 * self.box_height - 0.01
    level1_objs = list(filter(lambda o: level1_threshold < o.getZPosition() < level2_threshold, self.objects))
    level2_objs = list(filter(lambda o: level2_threshold < o.getZPosition() < level3_threshold, self.objects))
    level3_objs = list(filter(lambda o: level3_threshold < o.getZPosition() < level4_threshold, self.objects))
    return level1_objs, level2_objs, level3_objs

  def getNEachLevel(self):
    level1_objs, level2_objs, level3_objs = self.getObjEachLevel()
    n_level1 = len(level1_objs)
    n_level2 = len(level2_objs)
    n_level3 = len(level3_objs)
    return n_level1, n_level2, n_level3

  def checkRzValid(self):
    level1_objs, level2_objs, level3_objs = self.getObjEachLevel()
    level1_rz = list(map(lambda o: transformations.euler_from_quaternion(o.getRotation())[2], level1_objs))
    level2_rz = list(map(lambda o: transformations.euler_from_quaternion(o.getRotation())[2], level2_objs))
    level3_rz = list(map(lambda o: transformations.euler_from_quaternion(o.getRotation())[2], level3_objs))
    level1_rz_goal = self.pallet_rz
    if level1_rz_goal > np.pi:
      level1_rz_goal -= np.pi
    level2_rz_goal = self.pallet_rz + np.pi / 2
    if level2_rz_goal > np.pi:
      level2_rz_goal -= np.pi
    level3_rz_goal = level1_rz_goal

    def rz_close(rz, goal):
      angle_diff = abs(rz - goal)
      angle_diff = min(angle_diff, abs(angle_diff - np.pi))
      return angle_diff < np.deg2rad(20)

    level1_rz_ok = all(map(lambda rz: rz_close(rz, level1_rz_goal), level1_rz))
    level2_rz_ok = all(map(lambda rz: rz_close(rz, level2_rz_goal), level2_rz))
    level3_rz_ok = all(map(lambda rz: rz_close(rz, level3_rz_goal), level3_rz))
    return level1_rz_ok and level2_rz_ok and level3_rz_ok

def createBoxPalletizingEnv(config):
  return BoxPalletizingEnv(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 40, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 18, 'random_orientation': True,
                'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'fast', 'hard_reset_freq': 1000, 'object_scale_range': (1, 1),
                'kuka_adjust_gripper_offset': 0.001,
                }

  planner_config = {'random_orientation': True, 'half_rotation': True}

  env = BoxPalletizingEnv(env_config)
  planner = BoxPalletizingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)