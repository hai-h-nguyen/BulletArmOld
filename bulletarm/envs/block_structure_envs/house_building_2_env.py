from copy import deepcopy
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

class HouseBuilding2Env(BaseEnv):
  '''Open loop house building 2 task.

  The robot needs to first place two cubic blocks adjacent to each other, then place a roof on top.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    self.num_class = config['num_objects'] * 2 - 1
    super(HouseBuilding2Env, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(constants.CUBE, 2, random_orientation=self.random_orientation)
        self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
      except NoValidPositionException:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    top_blocks = []
    for block in blocks:
      if self._isObjOnTop(block, blocks):
        top_blocks.append(block)
    if len(top_blocks) != 2:
      return False
    if self._checkOnTop(top_blocks[0], roofs[0]) and self._checkOnTop(top_blocks[1], roofs[0]):
      return True
    return False

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(HouseBuilding2Env, self).isSimValid()

  def get_true_abs_state(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    
    if not self._checkObjUpright(roofs[0]) or not BaseEnv.isSimValid(self):
      return self.num_class
    dist_blocks = self._getDistance(blocks[0], blocks[1]) 
    if self._checkOnTop(blocks[0], roofs[0]) and \
       self._checkOnTop(blocks[1], roofs[0]) and \
       self._checkInBetween(roofs[0], blocks[0], blocks[1]):
      return 0
    if self._isObjectHeld(roofs[0]) and \
       dist_blocks < 2.2 * self.max_block_size and \
       dist_blocks > 2.1 * self.max_block_size:
      return 1
    if self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(roofs[0]) and \
       not roofs[0].isTouching(blocks[0]) and \
       not roofs[0].isTouching(blocks[1]) and \
       dist_blocks < 2.2 * self.max_block_size and \
       dist_blocks > 2.1 * self.max_block_size and \
       not self._checkInBetween(roofs[0], blocks[0], blocks[1]):
      return 2
    if self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(roofs[0]) and \
       not roofs[0].isTouching(blocks[0]) and \
       not roofs[0].isTouching(blocks[1]):
      return 4
    if self._isObjectHeld(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(roofs[0]) and \
       not roofs[0].isTouching(blocks[1]):
      return 3
    if self._isObjectHeld(blocks[1]) and \
       self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(roofs[0]) and \
       not roofs[0].isTouching(blocks[0]):
      return 3      
    return self.num_class

def createHouseBuilding2Env(config):
  return HouseBuilding2Env(config)
