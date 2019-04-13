import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

import numpy as np
import time
def print_t(str_):
  print( "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + str(str_))
  
def timestamp():
  return time.strftime("%y%m%dd%Hh%Mm%S")

""" 
   Deep Q Network 
"""
class DQNAgent:
  def __init__(self):
    self.gamma = 0.9
    self.memory = []
    self.learning_rate = 0.0005
    self.network_input_shape = (1,)
    self.actions = np.array(["NO_OP", "ACCELERATION", "DECELERATION", "CAR_INDEX_LEFT", "CAR_INDEX_RIGHT", "CLEAR", "FULL_THROTTLE", "EMERGENCY_BRAKE", "GO_LEFT", "GO_RIGHT"])
    self.network = None
    return
  
  def baseline_network(self, weights=None):
    # This returns a tensor
    inputs = Input(shape=self.network_input_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(output_dim=120, activation='relu')(inputs)
    x = Dropout(0.15)(x)
    x = Dense(output_dim=120, activation='relu')(x)
    x = Dropout(0.15)(x)
    x = Dense(output_dim=120, activation='relu')(x)
    x = Dropout(0.15)(x)
    predictions = Dense(len(self.actions), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    opt = Adam(self.learning_rate)
    model.compile(loss='mse', optimizer=opt)
    
    if weights:
      model.load_weights(weights)
    return model
  
  def load_network_hdf5(self):
    pass # Implement loading a hdf5 network 
  
  def save_network_hdf5(self, filename):
    pass # Implement saveing a network to hdf5
  
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
    
  
  pass