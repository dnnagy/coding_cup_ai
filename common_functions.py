import time
import numpy as np
from matplotlib import pyplot as plt

## 24 hour format ##
def print_t(str_):
  print( "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + str(str_))
  
def timestamp():
  return time.strftime("%y%m%dd%Hh%Mm%S")