"""
Sources:
https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/
https://danieltakeshi.github.io/2016/12/01/going-deeper-into-reinforcement-learning-understanding-dqn/
https://github.com/maurock/snake-ga/blob/master/
"""
from common_functions import *
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

""" 
   Deep Q Network 
"""
class DQNAgent:
  def __init__(self):
    self.gamma = 0.9
    self.memory = []
    self.learning_rate = 0.0005
    self.network_input_shape = (1,)
    self.actions = ["NO_OP", "ACCELERATION", "DECELERATION", "CAR_INDEX_LEFT", "CAR_INDEX_RIGHT", "CLEAR", "FULL_THROTTLE", "EMERGENCY_BRAKE", "GO_LEFT", "GO_RIGHT"]
    self.network = self.baseline_network()
    return
  
  def baseline_network(self):
    # This returns a tensor
    inputs = Input(shape=self.network_input_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(32, activation='relu')(inputs)
    x = Dropout(0.15)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.15)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.15)(x)
    predictions = Dense(len(self.actions), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    opt = Adam(self.learning_rate)
    model.compile(loss='mse', optimizer=opt)
    return model
  
  def load_network_hdf5(self):
    pass # Implement loading a hdf5 network 
  
  def save_network_hdf5(self, filename):
    pass # Implement saveing a network to hdf5
  
  # During gameplay, we should store all the experiences to use them in 
  # the experience replay.
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
    
  def train_on_sample(self, state, action, reward, next_state, done):
    # Do a feedforward pass for the current state s to get predicted Q-values for all actions.
    Q_pred = self.network.predict(state)

    # Do a feedforward pass for the next state s' and calculate maximum over all network outputs max_{a'}Q(s',a').
    Q_next_pred = self.network.predict(next_state)

    # Set Q-value target for action a to r+gamma*max_{a'}Q(s',a')
    # (use the max calculated in step 2). For all other actions, set the Q-value target
    # to the same as originally returned from step 1, making the error 0 for those outputs.
    target = reward
    if not done:
      target = reward + self.gamma * np.amax(Q_next_pred)
    Q_pred[np.argmax(action)] = target # action is one-hot encoded 

    # Update the weights using backpropagation.
    self.network.fit(state, Q_pred, epochs=1, verbose=0)

  # Implement experience replay technique. Memory can be collected anywhere,
  # it should contain (state, action, next_state, done) entries.
  def train_on_memory(self, memory=None):
    if memory==None:
      memory = self.memory

    if len(memory)>256:
      minibatch = random.sample(memory, 256)
    else:
      minibatch = memory
    for state, action, reward, next_state, done in minibatch:
      self.train_on_sample(state, action, reward, next_state, done)
  pass