import os
import pybullet as pb
import copy
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.equipments.box_color import BoxColor
from helping_hands_rl_envs.simulators.pybullet.equipments.rack import Rack
from helping_hands_rl_envs.simulators.pybullet.objects.plate import PLACE_RY_OFFSET, PLACE_Z_OFFSET
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.shelf_bowl_stacking_planner import ShelfBowlStackingPlanner
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util

class CovidTestEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    # self.shelf = Shelf()
    # self.rack = Rack(n=self.num_obj+1, dist=0.05)
    self.object_init_space = np.array([[0.3, 0.7],
                                       [-0.4, 0],
                                       [0, 0.40]])
    self.plate_model_id = None
    self.place_offset = None
    self.place_ry_offset = None
    self.end_effector_santilized_t = 0

    # for workspace = 0.4, flexible
    # rot90 x n
    R0 = np.array([[1., 0.],
                   [0., 1.]])
    R1 = np.array([[0., -1.],
                   [1., 0.]])
    R2 = np.array([[-1., 0.],
                   [0., -1.]])
    R3 = np.array([[0., 1.],
                   [-1., 0.]])
    self.rot90 = [R0, R1, R2, R3]
    # flip transform
    Nf = R0
    Vf = np.array([[-1., 0.],
                   [0., 1.]])
    self.flips = [Nf, Vf]

    self.box = BoxColor()
    self.new_tube_box_size = np.array([0.225, 0.225, 0.005])
    self.santilizing_box_size = np.array([0.225, 0.065, 0.03])
    self.used_tube_box_size = np.array([0.225, 0.095, 0.02])
    self.test_box_size = np.array([0.165, 0.395, 0.01])

    workspace_x = self.workspace[0, 1] - self.workspace[0, 0]
    workspace_y = self.workspace[1, 1] - self.workspace[1, 0]
    center_x = (self.workspace[0, 0] + self.workspace[0, 1]) / 2
    center_y = (self.workspace[1, 0] + self.workspace[1, 1]) / 2
    self.workspace_center = np.array([center_x, center_y, 0])
    upper_x = -workspace_x / 2 + self.new_tube_box_size[0] / 2
    lower_x = workspace_x / 2 - self.test_box_size[0] / 2
    new_tube_box_y = workspace_y / 2 - self.new_tube_box_size[1] / 2
    used_tube_box_y = -workspace_y / 2 + self.used_tube_box_size[1] / 2
    santilizing_box_y = (new_tube_box_y - self.new_tube_box_size[1] / 2 +
                        used_tube_box_y + self.used_tube_box_size[1] / 2) / 2
    self.new_tube_box_pos = np.array([upper_x, new_tube_box_y, 0])
    self.used_tube_box_pos = np.array([upper_x, used_tube_box_y, 0])
    self.santilizing_box_pos = np.array([upper_x, santilizing_box_y, 0])
    self.test_box_pos = np.array([lower_x, 0, 0])

    self.tube_pos_candidate = np.array([[-self.new_tube_box_size[0]/3, self.new_tube_box_size[1]/4, 0.01],
                                       [                           0, self.new_tube_box_size[1]/4, 0.01],
                                       [ self.new_tube_box_size[0]/3, self.new_tube_box_size[1]/4, 0.01]])
    self.swab_pos_candidate = np.array([[-self.new_tube_box_size[0]/3, -self.new_tube_box_size[1]/6, 0.01],
                                       [                           0, -self.new_tube_box_size[1]/6, 0.01],
                                       [ self.new_tube_box_size[0]/3, -self.new_tube_box_size[1]/6, 0.01]])
    self.tube_pos_candidate += self.new_tube_box_pos
    self.swab_pos_candidate += self.new_tube_box_pos

    # # for workspace = 0.4
    # self.box = BoxColor()
    # self.new_tube_box_pos = [0.4, 0.075, 0]
    # self.new_tube_box_size = [0.24, 0.24, 0.005]
    # self.santilizing_box_pos = [0.4, -0.075, 0]
    # self.santilizing_box_size = [0.24, 0.05, 0.03]
    # self.used_tube_box_pos = [0.4, -0.15, 0]
    # self.used_tube_box_size = [0.24, 0.09, 0.02]
    # self.test_box_pos = [0.6, 0.00, 0]
    # self.test_box_size = [0.15, 0.39, 0.01]
    # self.tube_pos_candidate = [[(0.35, 0.15, 0.01)],
    #                            [(0.4, 0.15, 0.01)],
    #                            [(0.45, 0.15, 0.01)]]
    # self.swab_pos_candidate = [[(0.35, 0.08, 0.01)],
    #                            [(0.4, 0.08, 0.01)],
    #                            [(0.45, 0.08, 0.01)]]

    # # for workspace = np.asarray([[0.1, 0.7],
      #                         [-0.3, 0.3],
      #                         [0, 0.50]])
    # self.new_tube_box_pos = [0.22, 0.12, 0]
    # self.new_tube_box_size = [0.24, 0.36, 0.01]
    # self.santilizing_box_pos = [0.22, -0.11, 0]
    # self.santilizing_box_size = [0.24, 0.08, 0.05]
    # self.used_tube_box_pos = [0.22, -0.23, 0]
    # self.used_tube_box_size = [0.24, 0.14, 0.1]
    # self.test_box_pos = [0.52, 0.00, 0]
    # self.test_box_size = [0.32, 0.6, 0.015]
    # self.tube_pos_candidate = [[(0.16, 0.22, 0.01)],
    #                            [(0.22, 0.22, 0.01)],
    #                            [(0.28, 0.22, 0.01)]]
    # self.swab_pos_candidate = [[(0.16, 0.05, 0.01)],
    #                            [(0.22, 0.05, 0.01)],
    #                            [(0.28, 0.05, 0.01)]]

  def initialize(self):
    super().initialize()
    self.robot.gripper_joint_limit = [0, 0.15]
    pass

  def reset(self):
    ''''''
    # self.plate_model_id = np.random.choice([1, 2, 6, 7, 8, 9])
    # self.plate_model_id = 0
    # self.end_effector_santilized_t = 0
    # self.place_ry_offset = PLACE_RY_OFFSET[self.plate_model_id]
    # self.place_offset = PLACE_Z_OFFSET[self.plate_model_id]
    self.placed_swab = False
    self.resetted = True
    self.no_obj_split = True

    # initial boxs in D4 group: rot90 + reflection
    rot_n = np.random.randint(0,4)
    flip = np.random.randint(0,2)
    # rot_n = 0
    # flip = 0

    self.new_tube_box_pos[:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.new_tube_box_pos[:2].T)).T
    self.used_tube_box_pos[:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.used_tube_box_pos[:2].T)).T
    self.santilizing_box_pos[:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.santilizing_box_pos[:2].T)).T
    self.test_box_pos[:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.test_box_pos[:2].T)).T

    self.new_tube_box_pos += self.workspace_center
    self.used_tube_box_pos += self.workspace_center
    self.santilizing_box_pos += self.workspace_center
    self.test_box_pos += self.workspace_center

    self.box.initialize(pos=self.new_tube_box_pos, rot=pb.getQuaternionFromEuler((0,0,90*rot_n)),
                        size=self.new_tube_box_size, color=[0.9, 0.9, 1, 1])
    self.box.initialize(pos=self.test_box_pos, rot=pb.getQuaternionFromEuler((0,0,90*rot_n)),
                        size=self.test_box_size, color=[0.9, 0.9, 0.9, 0.6])
    self.box.initialize(pos=self.santilizing_box_pos, rot=pb.getQuaternionFromEuler((0,0,90*rot_n)),
                        size=self.santilizing_box_size, color=[0.5, 0.5, 0.5, 0.6])
    self.box.initialize(pos=self.used_tube_box_pos, rot=pb.getQuaternionFromEuler((0,0,90*rot_n)),
                        size=self.used_tube_box_size, color=[1, 1, 0.5, 1])

    self.tube_pos_candidate[:,:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.tube_pos_candidate[:,:2].T)).T
    self.swab_pos_candidate[:,:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.swab_pos_candidate[:,:2].T)).T

    self.tube_pos_candidate += self.workspace_center
    self.swab_pos_candidate += self.workspace_center

    while True:
      self.resetPybulletEnv()
      try:
        # self._generateShapes(constants.RANDOM_BLOCK, self.num_obj, random_orientation=self.random_orientation,
        #                      pos=[(0.3, 0.12, 0.12)])
        tube_rot = 1.57 + np.random.rand() - 0.5
        for i in range(3):
          self._generateShapes(constants.TEST_TUBE,
                               rot=[pb.getQuaternionFromEuler([0., 0., tube_rot])],
                               pos=[tuple(self.tube_pos_candidate[i])])
        swab_rot = -1.57 + np.random.rand() - 0.5
        for i in range(3):
          self._generateShapes(constants.SWAB,
                               rot=[pb.getQuaternionFromEuler([0., 0., swab_rot])],
                               pos=[tuple(self.swab_pos_candidate[i])])
      except NoValidPositionException:
        continue
      else:
        break

    self._getObservation()
    return self._getObservation()

  def step(self, action):
    self.takeAction(action)
    motion_primative, x, y, z, rot = self._decodeAction(action)
    if self.end_effector_santilized_t > 0:
      self.end_effector_santilized_t += 1
    elif motion_primative == 0 and \
            self.isObjInBox([x, y, z], self.santilizing_box_pos, self.santilizing_box_size):
      self.end_effector_santilized_t = 1
    on_table_obj, on_table_obj_type = self.OnTableObj()
    on_table_tube = None
    on_table_swab = None
    if constants.TEST_TUBE in on_table_obj_type \
      and constants.SWAB in on_table_obj_type:
      for obj in on_table_obj:
        if obj.object_type_id == constants.SWAB:
          obj_rot_ = pb.getQuaternionFromEuler([0, 0, 0])
          self.objects.remove(obj)
          pb.removeBody(obj.object_id)
          # obj.resetPose([0.22, 0.06, 0.1], obj_rot_)
        if obj.object_type_id == constants.TEST_TUBE:
          rot = 2 * np.pi * np.random.rand()
          x_offset = (self.test_box_size[0] - 0.08) * np.random.rand()\
                     - (self.test_box_size[0] - 0.08) / 2
          y_offset = (self.test_box_size[1] - 0.08) * np.random.rand()\
                     - (self.test_box_size[1] - 0.08) / 2
          obj_rot_ = pb.getQuaternionFromEuler([0, 0, rot])
          obj.resetPose([0.6 + x_offset, 0. + y_offset, 0.05], obj_rot_)

        self.wait(20)
        self.placed_swab = True
      # for obj in on_table_obj:
      #   if obj.object_type_id == constants.SWAB:
      #     on_table_swab = obj
      #   elif obj.object_type_id == constants.TEST_TUBE:
      #     on_table_tube = obj
      #   if on_table_tube is not None and on_table_swab is not None:
      #     obj_pos, obj_rot = on_table_tube.getPose()
      #     oTtube = pybullet_util.getMatrix(obj_pos, obj_rot)
      #     swab_rot = pb.getQuaternionFromEuler([0, 0, 3.14])
      #     tubeTswab = pybullet_util.getMatrix([0.04, 0, 0], swab_rot)
      #     oTswab = oTtube.dot(tubeTswab)
      #     obj_pos_ = oTswab[:3, -1]
      #     obj_rot_ = transformations.quaternion_from_matrix(oTswab)
      #     on_table_swab.resetPose(obj_pos_, obj_rot_)

    self.wait(100)
    obs = self._getObservation(action)
    done = self._checkTermination()
    if done:
      reward = 1
    else:
      reward = 0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def isObjInBox(self, obj_pos, box_pos, box_size):
    box_range = self.box_range(box_pos, box_size)
    return box_range[0][0] < obj_pos[0] < box_range[0][1] and box_range[1][0] < obj_pos[1] < box_range[1][1]

  @staticmethod
  def box_range(box_pos, box_size):
    return np.array([[box_pos[0] - box_size[0] / 2, box_pos[0] + box_size[0] / 2],
                     [box_pos[1] - box_size[1] / 2, box_pos[1] + box_size[1] / 2]])

  # def isObjInUsedTubeBox(self, obj):
  #   return self.used_tube_box_range[0][0] < obj.getPosition()[0] < self.used_tube_box_range[0][1]\
  #          and self.used_tube_box_range[1][0] < obj.getPosition()[1] < self.used_tube_box_range[1][1]
  #
  # def getPlaceRyOffset(self):
  #   return PLACE_RY_OFFSET[self.plate_model_id]
  #
  # def anyObjectOnTarget1(self):
  #   for obj in self.objects:
  #     if self.shelf.isObjectOnTarget1(obj):
  #       return True
  #   return False

  def _checkTermination(self):
    # prepare multiple swab-tube pair
    if len(self.getTubes()) == 0 and self.end_effector_santilized_t == 1:
      return True

    # # prepare one swab-tube pair
    # for obj in self.objects:
    #   if self.isObjInBox(obj.getPosition(), self.used_tube_box_pos, self.used_tube_box_size)\
    #           and self.object_types[obj] == constants.TEST_TUBE\
    #           and self.placed_swab\
    #           and self.end_effector_santilized_t == 1:
    #     return True
    #   else:
    #     continue
    return False

  def OnTableObj(self):
    obj_list = []
    obj_type_list = []
    for obj in self.objects:
      if self.isObjInBox(obj.getPosition(), self.test_box_pos, self.test_box_size):
        obj_list.append(obj)
        obj_type_list.append(self.object_types[obj])
      # if self._isObjOnGround(obj) \
      #         and 0.33 < obj.getPosition()[0] < 0.7:
      #   obj_list.append(obj)
      #   obj_type_list.append(self.object_types[obj])
      # elif self._isObjOnGround(obj) \
      #         and (0.33 > obj.getPosition()[0] \
      #               or obj.getPosition()[0] > 0.7):
      #   self.no_obj_split = False
    return obj_list, obj_type_list

  def InBoxObj(self, box_pos, box_size):
    obj_list = []
    for obj in self.objects:
      if self.isObjInBox(obj.getPosition(), box_pos, box_size):
        obj_list.append(obj)
    return obj_list

  def getSwabs(self):
    swabs = []
    for obj in self.InBoxObj(self.new_tube_box_pos, self.new_tube_box_size):
      if obj.object_type_id == constants.SWAB:
        swabs.append(obj)
    return swabs

  def getTubes(self):
    tubes = []
    for obj in self.InBoxObj(self.new_tube_box_pos, self.new_tube_box_size):
      if obj.object_type_id == constants.TEST_TUBE:
        tubes.append(obj)
    return tubes

def createCovidTestEnv(config):
  return CovidTestEnv(config)


if __name__ == '__main__':
  object_init_space = np.asarray([[0.3, 0.7],
                          [-0.4, 0.4],
                          [0, 0.40]])
  workspace_size = 0.4
  workspace = np.asarray([[0.5 - workspace_size / 2, 0.5 + workspace_size / 2],
                          [0 - workspace_size / 2, 0 + workspace_size / 2],
                          [0, 0 + workspace_size]])
  env_config = {'workspace': workspace, 'object_init_space': object_init_space, 'max_steps': 10, 'obs_size': 128,
                'render': True, 'fast_mode': True, 'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 9,
                'random_orientation': True, 'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False,
                'object_scale_range': (0.6, 0.6), 'robot': 'kuka', 'object_init_space_check': 'point', 'physics_mode': 'slow'}
  planner_config = {'random_orientation': True}

  env = CovidTestEnv(env_config)
  # planner = ShelfBowlStackingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  a = 1