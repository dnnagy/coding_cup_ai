import numpy as np
import time
def print_t(str_):
  print( "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + str(str_))
  
def timestamp():
  return time.strftime("%y%m%dd%Hh%Mm%S")
  
"""
  Implement a class that can generate a state-space for the DQN based on tick data from server
"""
class StateSpace:
  def __init__(self):
    self.viewArea = [
    "         ",
    "   XXX   ",
    "   XXX   ",
    "   XXX   ",
    "   XXX   ",
    "  XXXXX  ",
    " XXXXXXX ",
    " XXXXXXX ",
    " XXXCXXX ",
    "  XXXXX  ",
    "   XXX   ",
    "    X    ",
    "         "]
    
    """ THIS FUNCTION IS TESTED """
    def _viewCoordsFromViewArea():
      carCoords = None
      seenCoords = []
      for y in range(len(self.viewArea)):
        line = list(self.viewArea[y])
        for x in range(len(line)):
          if line[x]=='C':
            carCoords = np.array([x,y])
          if line[x]=='X':
            seenCoords.append([x,y])
      return np.array([[c[0] - carCoords[0], c[1]-carCoords[1]] for c in seenCoords])
    
    self.viewAreaCoords = _viewCoordsFromViewArea()
    self.directionCodes = {
      # 0 is for no car
      'UP': 1,
      'DOWN': 2,
      'LEFT': 3,
      'RIGHT': 4
    }
    self.myCar = None
    pass
  
  """
    Get the coordinates of seen points relative to the car coordinates
    THIS FUNCTION IS TESTED
  """
  def relativeViewCoordsForDirection(self, d):
    def rmatrix(phi):
      return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    
    def dotnround(M, x):
      return np.round(M.dot(x)).astype(int)
    
    
    if d == 'UP':
      return self.viewAreaCoords
    elif d == 'DOWN':
      rotated = []
      R = rmatrix(np.pi)
      for c in self.viewAreaCoords:
        rotated.append(dotnround(R, c))
      return np.array(rotated)
    elif d == 'LEFT':
      rotated = []
      R = rmatrix(-np.pi/2.0)
      for c in self.viewAreaCoords:
        rotated.append(dotnround(R, c))
      return np.array(rotated)
    elif d == 'RIGHT':
      rotated = []
      R = rmatrix(np.pi/2.0)
      for c in self.viewAreaCoords:
        rotated.append(dotnround(R, c))
      return np.array(rotated)
    else: 
      raise ValueError("Invalid direction: "+d)
  
  """
    Build the state of the system, based on the current tick stored in game
  """
  def stateFromTick(self, game):
    if len(game.tick_data['cars'])>0:
      self.myCar = [c for c in game.tick_data['cars'] if c['id']==game.car_id][0]
    else:
      raise RuntimeError("Could not find my car.")
    
    print_t("My car is {}".format(self.myCar))
    
    # This is the state of the system that should be returned
    self.state = []
    
    # Check if the point is off the map
    def isOffMap(point):
      if point[0]<0 or point[0]>=60 or point[1]<0 or point[1]>=60:
        return True
      return False
    
    # Check if a car is on the point
    def carOnPoint(point):
      if len(game.tick_data['cars'])>=2:
        for car in game.tick_data['cars']:
          if car['pos']['x'] == point[0] and car['pos']['y'] == point[1]:
            return {"status": True, "car": car}
        return {"status": False, "car": None}
      return {"status": False, "car": None}
    
    # Check if a pedestrian is on the point
    def pedestrianOnPoint(point):
      if len(game.tick_data['pedestrians'])>=2:
        for ped in game.tick_data['pedestrians']:
          if ped['pos']['x'] == point[0] and ped['pos']['y'] == point[1]:
            return {"status": True, "ped": ped}
        return {"status": False, "ped": None}
      return {"status": False, "ped": None}
    
    # Check if a passenger is on the point
    def passengerOnPoint(point):
      if len(game.tick_data['passengers'])>=2:
        for pas in game.tick_data['passengers']:
          if pas['pos']['x'] == point[0] and pas['pos']['y'] == point[1]:
            return {"status": True, "pas": pas}
        return {"status": False, "pas": None}
      return {"status": False, "pas": None}
    
    # Iterate over all points seen by car
    relSeenCoords = self.relativeViewCoordsForDirection(self.myCar['direction'])
    for k in range(len(relSeenCoords)):
      mapPoint = [relSeenCoords[k][0]+self.myCar['pos']['x'],
                  relSeenCoords[k][1]+self.myCar['pos']['y']]
      
      if isOffMap(mapPoint):
        n_feat=11 # number of features in each point is 11
        self.state.append([-1 for k in range(n_feat)])
        continue
        
      car = carOnPoint(mapPoint)
      carDir = 0 if car['status']==False else self.directionCodes[car['car']['direction']]
      carSpeed = -1 if car['status']==False else car['car']['speed']
      
      ped = pedestrianOnPoint(mapPoint)
      pedDir = 0 if ped['status']==False else self.directionCodes[ped['ped']['direction']]
      pedSpeed = -1 if ped['status']==False else ped['ped']['speed']
      
      pas = passengerOnPoint(mapPoint)
      pasId = -1 if pas['status']==False else pas['pas']['id']
      pasDestX = -1 if pas['status']==False else pas['pas']['dest_pos']['x']
      pasDestY = -1 if pas['status']==False else pas['pas']['dest_pos']['y']
      pasIsInMyCar = -1 if pas['status']==False else int(self.myCar['id']==pas['pas']['car_id'])
      
      stateOfPoint = [
        game.mapMatrix[mapPoint[0]][mapPoint[1]], # What is the color of this point
        int(car['status']), # Is there any car on this point?
        carDir,
        carSpeed,
        int(ped['status']),
        pedDir,
        pedSpeed,
        int(pas['status']),
        pasDestX/60.0, # these should be normalized to 1 for NN
        pasDestY/60.0, # these should be normalized to 1 for NN
        pasIsInMyCar
      ] 
      
      self.state.append(np.array(stateOfPoint))
      
    self.state=np.array(self.state).flatten()
    self.state=np.append(self.state, np.array([
      self.myCar['pos']['x']/60.0, # these should be normalized to 1 for NN
      self.myCar['pos']['y']/60.0, # these should be normalized to 1 for NN
      self.directionCodes[self.myCar['direction']],
      self.myCar['speed'],
      self.myCar['life']/100.0 # these should be normalized to 1 for NN
    ]))
    return self.state

"""
  Baseline state-space: go to a certain point, without dying.
"""
class StateSpaceBaseline:
  pass