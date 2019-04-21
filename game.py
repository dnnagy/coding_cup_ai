import time
import socket
import json
from common_functions import *

class CCGame:
  def __init__(self, log_ticks=False):
    self.log_ticks = log_ticks
    self.tickLogs = []
    self.tcp_ip = '31.46.64.35'
    self.tcp_port = 12323
    self.team_token = '1iVXOVZK7ldH5Kr6qYCEkZE6xpR0SXZJkyfQayrKfJ2e9S8xdeTjsV9oohjePSsUXFOcDnevsu918'
    self.buffersize = 1024
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    self.valid_commands = np.array(["NO_OP", "ACCELERATION", "DECELERATION", "CAR_INDEX_LEFT", "CAR_INDEX_RIGHT", "CLEAR", "FULL_THROTTLE", "EMERGENCY_BRAKE", "GO_LEFT", "GO_RIGHT"])
    
    # Build map 
    # Regi verzio:
    # mapstr = "GPSSPGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGPSSPGPPSSPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPSSPPSSSSSSSSSSSSSSSSSSSSSSSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSPPSSPPPZZPPPPPPPPPPPPPPPPPPPPSSPPPPPPPPPPPPPPPPPPPPZZPPPSSPPGPSSPGPSSPGBBGBBGGBBGGBBGBBGPSSPGBBGBBGGBBGGBBGBBGPSSPBPSSPGGPSSPBPSSPPPPPPPPPPPPPPPPPPPPSSPPPPPPPPPPPPPPPPPPPPSSPBPSSPGGPSSPBPSSSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPGPSSPGGPSSPBPSSSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPBPSSPGGPSSPBPSSPPPPPPPSSPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPSSPBPSSPGGPSSPGPSSSPPGBBPSSPGGGGGBBBBBGPPPPPPPPGGBBBBBBGBBPSSSPGPSSPGGPSSPBPSSSSPPGBPSSPGTGPPPPPPPPPSSSSSSPPPBBBBBBGBBPSGSPBPSSPGGPSSPBPPSSSSPPGPSSPGGGPSSSSSSSSSSSSSSSSPPPGGGPPPPPSGSPBPSSPGGPSSPGGPPSSSSPPPSSPPBBPSSSSSSSSSSPPSSSSSSPPPGPSSSSSGSPGPSSPGGPSSPGGGPPSSSSPSSSSPPPPSSPSSPPSSPPPPPSSSSSSPPPSSSSSSSPGPSSPGGPSSPGGGGPPSSSSSBBSSSSSSSSSSPPSSSSSSPPPSSSSSSZSSPPPPPPGPSSPGGPSSPGGGGGPPSSSSBBSSSSSSSSSSPPSPPPPSSSPPPSSSSZSSPPSSPGGPSSPGGPSSPGGGGGGPPPPSSSSPPPPPPPPPPPSPPSSSSSSPPPPPPPSSPPSSPTTPSSPGGPSSPGGGGGGGGGPPSSPPSSSSSSSSGPSSSSPPSSSSPPGTGPSSSSSSPTGPSSPGGPSSPGGGGGGBBBGPSSSSSSSSSSSSSPPPPPPPPSSSSPPGPPSSSSSSPTTPSSPGGPSSPGGGGGGBBBGPSSSSSSPPPPSSSSSSSSSSPPSSSSPPPSSSPPPPPTTPSSPGGPSSPGGGGGGBBBGPSSPPPPPGGPPSSSSSSSSSSPPSSSSPSSSSPBBBBTGPSSPGGPSSPGPPPPPPPPPPSSPGGGGGGGPPPPPPPPSSSSPPSSSSSSSPPBBBBBGPSSPGGPSSPGPSSSSSSSSSSSPGGGGGGGGGGGBBGPPSSSSPPSSSSSPPGGGGGGGPSSPGGPSSPGPSSSSSSSSSSSPGGGGBGGGGGGBBGGPPSSSSPPPPSSSPPPPPPGTPSSPGGPSSPGPSSPPPPPPPSSPGGGGGGGGGGGGGGGGPPSSSSSSPSSSSSSSSPGTPSSPGGPSSPGPSSPBGBBBPSSPGGGGGGGGGGPPPPGGGPPSSSSSPPSSSSSSSPGTPSSPGGPSSPGPSSPBGBBBPSSPGGGGGGGGGPPSSPPGGGPPPPPPPPPPPPPZZPPGPSSPGGPZZPPPZZPPPPPPPSSPPPPPPPPPPPSSSSPPPPPPPPPPPPPPPPSSSSPPPZZPGGPSSSSSSSSSSSSSSSSSSSSSSSSSSSSBBSSSSSSSSSSSSSSSSSSBBSSSSSSPGGPSSSSSSSSSSSSSSSSSSSSSSSSSSSSBBSSSSSSSSSSSSSSSSSSBBSSSSSSPGGPZZPPPPPPPPPPPPPPPPSSPPPPPPPSSSSPPPPPPPSSPPPPPPPSSSSPPPZZPGGPSSPGPSSSSPPBBBBPPSSSPGGPPPPPSSPPPPPGGPSSSPPGBBPPSSPPGPSSPGGPSSPBPSSSSSPBBBBPSSSSPBBPSSSSSSSSSSPGGPSSSSPGBBGPSSPGBPSSPGGPSSPBPPPSSSPGGGGPSSSPPGGPSSSSSSSSSSPGGPPSSSPGGGGPSSPGBPSSPGGPSSPGGGPPSSPGGGGPSSPPGGGPSSPPPPPPSSPGGGPPSSPGGGGPSSPGBPSSPGGPSSPGGBBPSSPGGGGPSSPGGGBPSSPGGGGPSSPGGGGPSSPGGGGPSSPGBPSSPGGPSSPGGBBPSSPGGGGPSSPGBBBPSSPGGGGPSSPGGGGPSSPGGGGPSSPGBPSSPGGPSSPGPPPPSSPPPPPPZZPPPPPPZZPPPPPPSSPPPPPPZZPPPPPPZZPPGPSSPGGPSSPBPSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSZSSZSSSSSSSSPBPSSPGGPSSPBPSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSZSSZSSSSSSSSPBPSSPGGPSSPGPSSPPPPPSSPPPPPPPPPPPPZZPPPPPZZPPPPPSSPPPPPSSPPPBPSSPGGPSSPBPSSPGGGPSSPGGGGGGGBBBPSSPGGGPSSPBBGPSSPGBBPSSPBGBPSSPGGPSSPBPSSPGGGPSSPGGGGGGGBBBPSSPGGGPSSPBBGPSSPGBBPSSPBGBPSSPGGPSSPGPSSPPPPPZZPPPPPPPPPPPPSSPPPPPSSPPPPPSSPPPPPSSPPGGPSSPGGPSSPGPSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPGBPSSPGGPSSPGPSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPGBPSSPGGPSSPGPPSSPPPPZZPPPPSSPPPPZZPPPPSSPPPPPPPPPPSSPPPPZZPGBPSSPGGPSSPGGPSSPBBPSSPBBPSSPGBPSSPBGPSSPGGGGGGGGPSSPGGPSSPGBPSSPGGPSSPGGPSSPBBPSSPBBPSSPBBPSSPBGPSSPGGGGGGGGPSSPGGPSSPGBPSSPGGPSSPGGPSSPPPPSSPPPPZZPPPPSSPPPPZZPPPPPPPPPPZZPPPPSSPGGPSSPGGPSSPGGPSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPPPGGPSSPGGPSSPGGPSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPGGGGPSSPGGPSSPGGPPPPPPPPPPPPPPPPPPPPPPSSPPPPPPPPPPPPPPPPPPPPGGGGPSSPGGPSSPGGGGGGGGGGGGGBBBBGGBBBBPSSPBBBBGGBBBBGGGGGGGGGGGGGPSSPGPPSSPPPPPPPPPPPPPPPPPPPPPPPPPSSPPPPPPPPPPPPPPPPPPPPPPPPPSSPPSSSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSZSSSSSSSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSZSSSSPPSSPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPSSPPGPSSPGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGPSSPG"
    # Uj verzio:
    mapstr = "GPSSPRGGGGGTTTGGGGGGTTTGTTGGGGGGTTTGTGGGGGGGTTGGGGGGGGRPSSPGPPSSPCPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPCPSSPPSSSSSXSSSSSSSSSSSSSSSSSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSXSSSSSSSSSSXSSSSSSSSSSSSSSSSSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSXSSSSSPPSSPCPZZPPPPPPPPPPPPPPPPPPPPSSPPPPPPPPPPPPPPPPPPPPZZPCPSSPPRCXXCRCXXCRRRRRRRRRRRRRRRRRRCXXCRRRRRRRRRRRRRRRRRRCXXCRCXXCRTPSSPRPSSPPPPPPPPPPPPPPPPPPPPSSPPPPPPPPPPPPPPPPPPPPSSPRPSSPGTPSSPRPSSSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPRPSSPGGPSSPRPSSSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPRPSSPTTPSSPRPSSPPPPPPPSSPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPSSPRPSSPGTPSSPRPSSSPPTBBPSSPGTGGGBBBBBGPPPPPPPPGTBBBBBBGBBPSSSPRPSSPGGPSSPRPSSSSPPTBPSSPTTTPPPPPPPPPSSSSSSPPPBBBBBBGBBPSGSPRPSSPTGPSSPRPPSSSSPPTPSSPGTGPSSSSSSSSSSSSSSSSPPPGTTPPPPPSGSPRPSSPTGPSSPRGPPSSSSPPPSSPPBBPSSSSSSSSSSPPSSSSSSPPPGPSSSSSGSPRPSSPGTPSSPRGGPPSSSSPSSSSPPPPSSPSSPPSSPPPPPSSSSSSPPPSSSSSSSPRPSSPGTPSSPRGGGPPSSSSSBBSSSSSSSSSSPPSSSSSSPPPSSSSSSZSSPPPPPPRPSSPTGPSSPRGGGGPPSSSSBBSSSSSSSSSSPPSPPPPSSSPPPSSSSZSSPPSSPGRPSSPTTPSSPRGGGGGPPPPSSSSPPPPPPPPPPPSPPSSSSSSPPPPPPPSSPPSSPTRPSSPGGPSSPRGGGGGGGGPPSSPPSSSSSSSSGPSSSSPPSSSSPPGTGPSSSSSSPTRPSSPGGPSSPRGGGGGBBBGPSSSSSSSSSSSSSPPPPPPPPSSSSPPGPPSSSSSSPTRPSSPGGPSSPRGGGGGBBBGPSSSSSSPPPPSSSSSSSSSSPPSSSSPPPSSSPPPPPTRPSSPTGPSSPRGGGGGBBBGPSSPPPPPGGPPSSSSSSSSSSPPSSSSPSSSSPBBBBTRPSSPGGPSSPRPPPPPPPPPPSSPGGTTTTTPPPPPPPPSSSSPPSSSSSSSPPBBBBBRPSSPGGPSSPRPSSSSSSSSSSSPGGTGGGTTTTGBBGPPSSSSPPSSSSSPPGGGGGTRPSSPGTPSSPRPSSSSSSSSSSSPGGTTBGTTTTTBBTGPPSSSSPPPPSSSPPPPPPTRPSSPGTPSSPRPSSPPPPPPPSSPGGTTGGTTGGTGTTTGPPSSSSSSPSSSSSSSSPGRPSSPTTPSSPRPSSPBGBBBPSSPGGTTGTTTGGPPPPTGGPPSSSSSPPSSSSSSSPGRPSSPTTPSSPRPSSPBGBBBPSSPGGGGGGGGGPPSSPPGGGPPPPPPPPPPPPPZZPPRPSSPGGPZZPCPZZPPPPPPPSSPPPPPPPPPPPSSSSPPPPPPPPPPPPPPPPSSSSPCPZZPGGPSSSXSSSSSSSSSSSSSSSSSSSSSSSSBBSSSSSSSSSSSSSSSSSSBBSSXSSSPTGPSSSXSSSSSSSSSSSSSSSSSSSSSSSSBBSSSSSSSSSSSSSSSSSSBBSSXSSSPTGPZZPCPPPPPPPPPPPPPPSSPPPPPPPSSSSPPPPPPPSSPPPPPPPSSSSPCPZZPGGPSSPRPSSSSPPBBBBPPSSSPGGPPPPPSSPPPPPGGPSSSPPGBBPPSSPPRPSSPTTPSSPRPSSSSSPBBBBPSSSSPBBPSSSSSSSSSSPGGPSSSSPGBBGPSSPTRPSSPTTPSSPRPPPSSSPGGGGPSSSPPGGPSSSSSSSSSSPGTPPSSSPTGTGPSSPTRPSSPGTPSSPRTTPPSSPGTTGPSSPPTTGPSSPPPPPPSSPGTGPPSSPGTTGPSSPTRPSSPGGPSSPRGBBPSSPGTTTPSSPGTTBPSSPGGGTPSSPGTTGPSSPGTTTPSSPGRPSSPGGPSSPRGBBPSSPGTTTPSSPGBBBPSSPGGTTPSSPGGGGPSSPGGTTPSSPGRPSSPGTPSSPRPPPPSSPPPPPPZZPPPPPPZZPPPPPPSSPPPPPPZZPPPPPPZZPPRPSSPGTPSSPRPSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSZSSZSSSSSSSSPRPSSPGGPSSPRPSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSZSSZSSSSSSSSPRPSSPGGPSSPRPSSPPPPPSSPPPPPPPPPPPPZZPPPPPZZPPPPPSSPPPPPSSPPPRPSSPGGPSSPRPSSPTTGPSSPGGGTTTGBBBPSSPGTTPSSPBBGPSSPGBBPSSPBGRPSSPTGPSSPRPSSPGGGPSSPGGGGTTGBBBPSSPGTGPSSPBBTPSSPTBBPSSPBTRPSSPTGPSSPRPSSPPPPPZZPPPPPPPPPPPPSSPPPPPSSPPPPPSSPPPPPSSPPGRPSSPGGPSSPRPSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPGRPSSPGGPSSPRPSSSSSSZSSZSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPTRPSSPTTPSSPRPPSSPPPPZZPPPPSSPPPPZZPPPPSSPPPPPPPPPPSSPPPPZZPTRPSSPGTPSSPRGPSSPBBPSSPBBPSSPGBPSSPBGPSSPGTTTGGGGPSSPGTPSSPGRPSSPTTPSSPRGPSSPBBPSSPBBPSSPBBPSSPBGPSSPTTTTGTTGPSSPGGPSSPGRPSSPGGPSSPRGPSSPPPPSSPPPPZZPPPPSSPPPPZZPPPPPPPPPPZZPPPPSSPTRPSSPTGPSSPRTPSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPPPGRPSSPGGPSSPRTPSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSPGGTRPSSPGGPSSPRTPPPPPPPPPPPPPPPPPPPPPPSSPPPPPPPPPPPPPPPPPPPPGTTRPSSPGRCXXCRRRRRRRRRRRRRRRRRRRRRRRCXXCRRRRRRRRRRRRRRRRRRRRRRRCXXCRPPSSPCPPPPPPPPPPPPPPPPPPPPPPPSSPPPPPPPPPPPPPPPPPPPPPPPCPSSPPSSSSZXSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSXZSSSSSSSSZXSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSXZSSSSPPSSPCPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPCPSSPPGPSSPRGGGGTTGGGTGGGGGGTTGGGGGGGGGGGTTGGGGGTGGTTGGGTTGGRPSSPG"
    mapstrToInt = {
      'S': 0, # Aszfaltozott út
      'Z': 1, # Zebra
      'P': 2, # Járda
      'G': 3, # fű, zöld terület
      'B': 4, # épület
      'T': 5, # fa
      'R': 6, # szabadpályás vasút
      'C': 7, # vasúti gyalogos átjáró
      'X': 8, # vasúti autós átjáró
    }
    self.mapMatrix = np.array([mapstrToInt[x] for x in list(mapstr)]).reshape(60,60)
    
    self.started = False
    self.tick_data_prev = None # Previous tick data
    self.tick_data = None # Current tick data 
    
    self.game_id = None
    self.ticknum = None
    self.car_id = None
    self.myCar_prev = None # Previous car object
    self.myCar = None # Current car object
    self.crashed = False
    return
    
  def get_response(self):
    if not self.started:
      raise RuntimeError("Game not started yet.")
    
    # Update tick_data and tick_data_prev
    if self.tick_data != None:
      self.tick_data_prev = self.tick_data
    self.tick_data = json.loads(self.socket.recv(self.buffersize).decode())
    if self.log_ticks==True:
      self.tickLogs.append(self.tick_data)
        
    if len(self.tick_data['messages']) > 0:
      if 'END REASON: CRASHED' in self.tick_data['messages']:
        self.crashed = True
        self.close()
        print_t('CAR CRASHED.')
      else:
        raise ValueError("Messages array is not empty:\n {}".format(self.tick_data['messages']))
    
    
    # Update myCar and myCar_prev
    if self.myCar != None:
      self.myCar_prev = self.myCar
      
    if len(self.tick_data['cars'])>0 and self.car_id!=None:
      self.myCar = [c for c in self.tick_data['cars'] if c['id']==self.car_id][0]
      
    self.game_id = self.tick_data['request_id']['game_id']
    self.ticknum = self.tick_data['request_id']['tick']
    self.car_id = self.tick_data['request_id']['car_id']
    return
  
  def start(self):
    if self.started == True:
      raise RuntimeError("Game already started.")
      
    self.socket.connect((self.tcp_ip, self.tcp_port))
    self.started = True
    self.tickLogs = []
    
    # Send first messsage
    first_message = {
      "token": self.team_token
    }
    first_message = json.dumps(first_message).encode()
    
    self.socket.send(first_message)
    self.get_response()
    return
    
  def close(self):
    if self.started == False:
      print_t("Game already closed.")
    self.socket.close()
    self.started = False
    if self.log_ticks==True:
      pass # Save tick logs to a file
    return
  
  def draw_map(self):
    plt.figure(figsize=(8,8), dpi=112)
    plt.imshow(self.mapMatrix)
    return
  
  def send_random_command(self):
    if self.started == False:
      raise RuntimeError("Game is stopped.")
    cmd = np.random.choice(self.valid_commands)
    msg = {"response_id":
           {"game_id": self.game_id, "tick": self.ticknum, "car_id": self.car_id},
           "command": cmd
          }
    self.socket.send(json.dumps(msg).encode())
    self.get_response()
    return cmd
    
  def send_command(self, command):
    if self.started == False:
      raise RuntimeError("Game is stopped.")
    msg = {"response_id":
           {"game_id": self.game_id, "tick": self.ticknum, "car_id": self.car_id},
           "command": command
          }
    self.socket.send(json.dumps(msg).encode())
    self.get_response()
    return