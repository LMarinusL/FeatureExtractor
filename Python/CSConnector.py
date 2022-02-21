import socket
import time
import pandas as pd
import io 
#print(pd.__version__)  

# env: python 3.8.5 - py
# env: python 3.10 - pip
# py -m pip install pandas

# https://github.com/CanYouCatchMe01/CSharp-and-Python-continuous-communication

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
    print("data recieved")
    break
print(pd.__version__)  
df = pd.read_csv(io.StringIO(receivedData), sep=" ")
#df = pd.read_csv(receivedData, delim_whitespace=False, header=0)
print(df.keys())