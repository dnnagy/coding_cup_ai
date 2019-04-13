import numpy as np
import time
def print_t(str_):
  print( "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + str(str_))
  
def timestamp():
  return time.strftime("%y%m%dd%Hh%Mm%S")

"""
  Implement a class that rewards the DQN agent
"""
class Rewarder:
  def __init__(self):
    pass
  
  def calculate_reward(self, game):
    rew = 0
    if game.myCar != None and game.myCar_prev != None:
      lifeDiff = game.myCar_prev['life'] - game.myCar['life']
      rew = rew-lifeDiff
      
      transportedDiff = game.myCar['transported'] - game.myCar_prev['transported'] 
      rew = rew + 200*transportedDiff
      
      if ('passenger_id' in game.myCar) and not ('passenger_id' in game.myCar_prev):
        rew = rew + 100
    return rew

"""
  Baseline rewarder: go to a certain point, without dying.
"""
class RewarderBaseline:
  pass