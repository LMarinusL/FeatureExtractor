import socket
import time
import pandas as pd

# env: 'base': conda

# https://github.com/CanYouCatchMe01/CSharp-and-Python-continuous-communication
#print(pd.__version__)

host, port = "127.0.0.1", 25001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

startTime = 0
while True:
    time.sleep(1) 
    startTime +=1 
    timeString = str(startTime)
    print(timeString)

    sock.sendall(timeString.encode("UTF-8")) 
    receivedData = sock.recv(1024).decode("UTF-8") 
    print(receivedData)
    
df = pd.read_csv(receivedData, delim_whitespace=False, header=0)
print(df.keys())