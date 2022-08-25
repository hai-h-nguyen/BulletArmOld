from copy import deepcopy
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.envs.utils.check_goal import CheckGoal
from bulletarm.envs.utils.check_goal_custom_labels import CheckGoalCustomLabels
from bulletarm.pybullet.utils.constants import NoValidPositionException

class HouseBuildingXEnv(BaseEnv):
  ''''''
  def __init__(self, config):
    self.goal = config["goal_string"]
    self.check_goal = CheckGoal(self.goal, self)
    self.check_goal.parse_goal_()
    config['num_objects'] = self.check_goal.num_objects
    self.num_class = config['num_objects'] * 2 - 1
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(HouseBuildingXEnv, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
      try:
        # order matters!
        self._generateShapes(constants.ROOF, self.check_goal.num_roofs, random_orientation=self.random_orientation)
        self._generateShapes(constants.BRICK, self.check_goal.num_bricks, random_orientation=self.random_orientation)
        self._generateShapes(constants.CUBE, self.check_goal.num_blocks, random_orientation=self.random_orientation)
        self._generateShapes(constants.TRIANGLE, self.check_goal.num_triangles, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    if self.goal in ['1b1b1r', '1b1r']:
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkStack(blocks+triangles) and self._checkObjUpright(triangles[0])

    elif self.goal in ['1l1l2r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkStack(bricks+roofs) and self._checkObjUpright(roofs[0])

    elif self.goal in ['2b1l1r']:
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return self._checkOnTop(blocks[0], bricks[0]) and \
            self._checkOnTop(blocks[1], bricks[0]) and \
            self._checkOnTop(bricks[0], triangles[0]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._checkInBetween(triangles[0], blocks[0], blocks[1])

    elif self.goal in ['1l1b1r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkStack(bricks+blocks+triangles) and self._checkObjUpright(triangles[0])

    elif self.goal in ['1l2b1r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            (self._checkOnTop(blocks[0], triangles[0]) or self._checkOnTop(blocks[1], triangles[0])) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1])

    elif self.goal in ['1l2b2r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkOnTop(blocks[0], roofs[0]) and \
            self._checkOnTop(blocks[1], roofs[0]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1])

    elif self.goal in ['1l1l1r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkStack(bricks+triangles) and self._checkObjUpright(triangles[0])

    elif self.goal == '2b2r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(blocks[0], roofs[0]) and \
            self._checkOnTop(blocks[1], roofs[0]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1])

    elif self.goal == '2b1r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return (self._checkOnTop(blocks[0], triangles[0]) or self._checkOnTop(blocks[1], triangles[0])) and \
            self._checkAdjacent(blocks[0], blocks[1])

    elif self.goal == '2b1b1r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return (self._checkStack(blocks[1:]+triangles) or self._checkStack([blocks[0]] + blocks[2:] + triangles)) and \
            self._checkAdjacent(blocks[0], blocks[1])

    elif self.goal == '2b2b1r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return self._checkAdjacent(blocks[0], blocks[1]) and \
            self._checkOnTop(blocks[0], blocks[2]) and \
            self._checkOnTop(blocks[1], blocks[3]) and \
            (self._checkOnTop(blocks[2], triangles[0]) or self._checkOnTop(blocks[3], triangles[0]))

    elif self.goal == '2b2b2r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(blocks[0], blocks[2]) and \
            self._checkOnTop(blocks[1], blocks[3]) and \
            self._checkOnTop(blocks[3], roofs[0]) and \
            self._checkOnTop(blocks[2], roofs[0]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1])

    elif self.goal == '1l1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return self._checkOnTop(bricks[0], triangles[0])

    elif self.goal == '1l2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(bricks[0], roofs[0])

    elif self.goal == '2b1l2r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(blocks[0], bricks[0]) and \
            self._checkOnTop(blocks[1], bricks[0]) and \
            self._checkOnTop(bricks[0], roofs[0]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1])

  def getObjectPosition(self):
    return list(map(self._getObjectPosition, self.objects))

  def isSimValid(self):
    if self.goal in ['1l1l1r', '1l1b1r', '2b1r', '2b1b1r', '1l1r', '2b1l1r', '1l2b1r', '1b1r', '1b1b1r', '2b2b1r']:
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkObjUpright(triangles[0]) and super(HouseBuildingXEnv, self).isSimValid()

    elif self.goal in ['2b2r', '2b2b2r', '1l2r', '1l1l2r', '2b1l2r', '1l2b2r']:
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkObjUpright(roofs[0]) and super(HouseBuildingXEnv, self).isSimValid()

  def get_true_abs_state(self):
    if self.goal == '1l1b1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      if not self._checkObjUpright(triangles[0]) or not BaseEnv.isSimValid(self):
        return self.num_class

      if self._checkOnTop(blocks[0], triangles[0]) and \
          self._checkOnTop(bricks[0], blocks[0]) and \
          self._isObjOnGround(bricks[0]):
          return 0
      
      if self._checkOnTop(bricks[0], blocks[0]) and \
          self._isObjOnGround(bricks[0]) and \
          self._isObjectHeld(triangles[0]):
          return 1

      if self._checkOnTop(bricks[0], blocks[0]) and \
          self._isObjOnGround(bricks[0]) and \
          self._isObjOnGround(triangles[0]) and \
          not triangles[0].isTouching(bricks[0]):
          return 2
      
      if self._isObjectHeld(blocks[0]) and \
          self._isObjOnGround(bricks[0]) and \
          self._isObjOnGround(triangles[0]) and \
          not triangles[0].isTouching(bricks[0]):
          return 3
      
      if self._isObjOnGround(bricks[0]) and \
          self._isObjOnGround(triangles[0]) and \
          self._isObjOnGround(blocks[0]) and \
          not triangles[0].isTouching(bricks[0]) and \
          not triangles[0].isTouching(blocks[0]) and \
          not bricks[0].isTouching(blocks[0]):
          return 4
      return self.num_class

    if self.goal == '1l2b1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      if not self._checkObjUpright(triangles[0]) or not BaseEnv.isSimValid(self):
        return self.num_class

      if self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            (self._checkOnTop(blocks[0], triangles[0]) or self._checkOnTop(blocks[1], triangles[0])) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]):
            return 0

      if self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._isObjectHeld(triangles[0]):
            return 1

      if self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            not triangles[0].isTouching(bricks[0]) and \
            self._isObjOnGround(triangles[0]):
            return 2
      
      for i in range(2):
        if self._isObjectHeld(blocks[i]) and \
            self._checkOnTop(bricks[0], blocks[1-i]) and \
            not triangles[0].isTouching(bricks[0]) and \
            self._isObjOnGround(triangles[0]):
            return 3

        if self._isObjOnGround(blocks[i]) and \
            self._checkOnTop(bricks[0], blocks[1-i]) and \
            self._isObjOnGround(triangles[0]) and \
            not triangles[0].isTouching(bricks[0]) and \
            not triangles[0].isTouching(blocks[i]):
            return 4
        
        if self._isObjectHeld(blocks[i]) and \
            self._isObjOnGround(blocks[1-i]) and \
            self._isObjOnGround(triangles[0]) and \
            not triangles[0].isTouching(bricks[0]) and \
            not triangles[0].isTouching(blocks[1-i]) and \
            not bricks[0].isTouching(blocks[1-i]):
            return 5

        if self._isObjOnGround(blocks[i]) and \
            self._isObjOnGround(blocks[1-i]) and \
            self._isObjOnGround(bricks[0]) and \
            self._isObjOnGround(triangles[0]) and \
            not bricks[0].isTouching(blocks[1-i]) and \
            not bricks[0].isTouching(blocks[i]) and \
            not bricks[0].isTouching(triangles[0]) and \
            not triangles[0].isTouching(blocks[1-i]) and \
            not triangles[0].isTouching(blocks[i]):
            return 6
      return self.num_class

    if self.goal == '1l2b2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      if not self._checkObjUpright(roofs[0]) or not BaseEnv.isSimValid(self):
        return self.num_class

      if self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkOnTop(blocks[0], roofs[0]) and \
            self._checkOnTop(blocks[1], roofs[0]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1]):
        return 0

      if self._isObjectHeld(roofs[0]) and \
            self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]):
        return 1

      if self._isObjOnGround(roofs[0]) and \
            not roofs[0].isTouching(blocks[0]) and \
            not roofs[0].isTouching(blocks[1]) and \
            not roofs[0].isTouching(bricks[0]) and \
            self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]):
        return 2

      for i in range(2):
        if self._isObjectHeld(blocks[i]) and \
            self._checkOnTop(bricks[0], blocks[1-i]) and \
            self._isObjOnGround(roofs[0]) and \
            not roofs[0].isTouching(bricks[0]) and \
            not roofs[0].isTouching(blocks[1-i]):
          return 3

        if self._isObjOnGround(blocks[i]) and \
            self._isObjOnGround(roofs[0]) and \
            self._checkOnTop(bricks[0], blocks[1-i]) and \
            not roofs[0].isTouching(bricks[0]) and \
            not roofs[0].isTouching(blocks[1-i]) and \
            not roofs[0].isTouching(blocks[i]):
          return 4

        if self._isObjectHeld(blocks[i]) and \
            self._isObjOnGround(roofs[0]) and \
            self._isObjOnGround(blocks[1-i]) and \
            self._isObjOnGround(bricks[0]) and \
            not roofs[0].isTouching(bricks[0]) and \
            not roofs[0].isTouching(blocks[1-i]) and \
            not bricks[0].isTouching(blocks[1-i]):
          return 5

        if self._isObjOnGround(blocks[i]) and \
            self._isObjOnGround(blocks[1-i]) and \
            self._isObjOnGround(bricks[0]) and \
            self._isObjOnGround(roofs[0]) and \
            not roofs[0].isTouching(bricks[0]) and \
            not roofs[0].isTouching(blocks[i]) and \
            not roofs[0].isTouching(blocks[1-i]) and \
            not bricks[0].isTouching(blocks[i]) and \
            not bricks[0].isTouching(blocks[1-i]) and \
            not blocks[i].isTouching(blocks[1-i]):
          return 6
      
      return self.num_class

    if self.goal == '1l1l1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
         
      if not self._checkObjUpright(triangles[0]) or not BaseEnv.isSimValid(self):
        return self.num_class
      
      if self._checkStack() and self._checkObjUpright(triangles[0]):
        return 0

      if self._checkStack(bricks) and self._isObjectHeld(triangles[0]):
        return 1

      if self._checkStack(bricks) and \
          self._isObjOnGround(triangles[0]) and \
          not triangles[0].isTouching(bricks[0]) and \
          not triangles[0].isTouching(bricks[1]):
          return 2

      for i in range(2):
        if self._isObjOnGround(triangles[0]) and \
            self._isObjOnGround(bricks[i]) and \
            self._isObjectHeld(bricks[1-i]) and \
            not triangles[0].isTouching(bricks[i]):
            return 3
        
        if self._isObjOnGround(triangles[0]) and \
            self._isObjOnGround(bricks[i]) and \
            self._isObjOnGround(bricks[1-i]) and \
            not triangles[0].isTouching(bricks[i]) and \
            not triangles[0].isTouching(bricks[1-i]) and \
            not bricks[1-i].isTouching(bricks[i]):
            return 4
      return self.num_class

    if self.goal == '1l1l2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      if not self._checkObjUpright(roofs[0]) or not BaseEnv.isSimValid(self):
        return self.num_class
      
      if self._checkStack() and self._checkObjUpright(roofs[0]):
        return 0

      if self._checkStack(bricks) and self._isObjectHeld(roofs[0]):
        return 1

      if self._checkStack(bricks) and \
          self._isObjOnGround(roofs[0]) and \
          not roofs[0].isTouching(bricks[0]) and \
          not roofs[0].isTouching(bricks[1]):
          return 2

      for i in range(2):
        if self._isObjOnGround(roofs[0]) and \
            self._isObjOnGround(bricks[i]) and \
            self._isObjectHeld(bricks[1-i]) and \
            not roofs[0].isTouching(bricks[i]):
            return 3
        
        if self._isObjOnGround(roofs[0]) and \
            self._isObjOnGround(bricks[i]) and \
            self._isObjOnGround(bricks[1-i]) and \
            not roofs[0].isTouching(bricks[i]) and \
            not roofs[0].isTouching(bricks[1-i]) and \
            not bricks[1-i].isTouching(bricks[i]):
            return 4
      return self.num_class

def createHouseBuildingXEnv(config):
  return HouseBuildingXEnv(config)
