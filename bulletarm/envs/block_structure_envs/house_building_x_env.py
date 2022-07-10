from copy import deepcopy
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.envs.utils.check_goal import CheckGoal
from bulletarm.envs.utils.check_goal_custom_labels import CheckGoalCustomLabels
from bulletarm.pybullet.utils.constants import NoValidPositionException

class HouseBuildingXEnv(BaseEnv):
  ''''''
  def __init__(self, config):
    super(HouseBuildingXEnv, self).__init__(config)

    self.goal = config["goal_string"]
    self.check_goal = CheckGoal(self.goal, self)

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

def createHouseBuildingXEnv(config):
  return HouseBuildingXEnv(config)
