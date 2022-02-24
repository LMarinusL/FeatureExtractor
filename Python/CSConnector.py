import socket
import time
import io 
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
print(np.__version__)
print(pd.__version__)
print(sys.version)
print(sklearn.__version__)
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
plt.rcParams.update({'font.size': 16})

#print(pd.__version__)  

# env: python 3.8.5 - py
# env: python 3.10 - pip
# py -m pip install pandas

# https://github.com/CanYouCatchMe01/CSharp-and-Python-continuous-communication
"""
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
print(df.head())
"""
##############
import pathlib
print(pathlib.Path(__file__).parent.resolve())
df = pd.read_csv('C:/Users/neder/Documents/Geomatics/Unity/PCproject/DEMViewer/Assets/Output/outputGridFull.txt', sep=" ")
print(df.head())
##############

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

################
col_study = ['year', 'interval', 'x', 'y', 'hprevious', 'slope', 'aspect', 'curvature']
param_study = 'hdifference'

###############

dfTrain = sklearn.utils.resample(df[df.year < 2017], n_samples=50000, random_state=None, stratify=None)
Xo = dfTrain[col_study]
yo = dfTrain[param_study]
dfTest = sklearn.utils.resample(df[df.year > 2017], n_samples=50000, random_state=None, stratify=None)
print('resampled')
Xt = dfTest[col_study]
yt = dfTest[param_study]
X_traino, X_testo, y_traino, y_testo = train_test_split(Xo, yo, test_size=0.3, random_state=42)
print('split')
forest2 = RandomForestRegressor()
forest2.fit(X_traino, y_traino)
print('fitted')
y_train_pred = forest2.predict(X_traino)
y_test_pred = forest2.predict(X_testo)
y_pred = forest2.predict(Xt)
print('predicted')
################
plt.rcParams.update({'font.size': 15})

result = pd.DataFrame(forest2.feature_importances_,  df[col_study].columns)
result.columns = ['importance']
result.sort_values(by='importance', ascending=False)

features=df.columns[[0,1,2,3,4,6,7,8,9]]

importances = forest2.feature_importances_
indices = np.argsort(importances)
fig1 = plt.figure()
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()
##################
###################
plt.rcParams.update({'font.size': 20})
fig2, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                                    figsize=(20, 10))
cm = plt.cm.get_cmap('RdYlBu')
sc = ax[0].scatter(Xt['x'], Xt['y'],
           linewidths=1, alpha=.7,
            edgecolor='none',
           s = 20,
           c=(yt),
            cmap=cm, vmin=-2, vmax=2)
ax[0].set_title('Actual')
ax[0].set_xlabel("x coordinate")
ax[0].set_ylabel("y coordinate")
ax[0].tick_params(labelsize=12)
cm = plt.cm.get_cmap('RdYlBu')
sc = ax[1].scatter(Xt['x'], Xt['y'],
           linewidths=1, alpha=.7,
            edgecolor='none',
           s = 20,
           c=(y_pred),
            cmap=cm, vmin=-2, vmax=2)
cbar = fig2.colorbar(sc)
cbar.ax.set_ylabel('Change in bed level height per year [m]', rotation=270)
cbar.ax.get_yaxis().labelpad = 20
ax[1].set_title('Prediction')
ax[1].set_xlabel("x coordinate")
ax[1].tick_params(labelsize=12)
sc = ax[2].scatter(Xt['x'], Xt['y'],
           linewidths=1, alpha=.7,
            edgecolor='none',
           s = 20,
           c=(y_pred - yt),
            cmap=cm,  vmin=-2, vmax=2)
ax[2].set_title('Residual')
ax[2].set_xlabel("x coordinate")
ax[2].tick_params(labelsize=12)

fig2.subplots_adjust(wspace=0.03, hspace=0)
fig2.suptitle('Annual sedimentation 2012-2018')
plt.show()

#######
print(" shown map residuals")
fig3 = plt.figure()
plt.title('2012-2018 Sedimentation')
plt.boxplot([y_pred, yt, (y_pred - yt)], showfliers=False)
plt.xticks([1, 2, 3], ['Prediction', 'Actual', 'Residuals'])
plt.ylabel('[m]')
plt.show()

###############

fig4 = plt.figure()
plt.figure(figsize=(12,8))
plt.title('Random Forest Regressor')
plt.scatter(y_train_pred, y_train_pred - y_traino, c='blue', marker='o',s=10, label='Training dataset')
plt.scatter(y_test_pred, y_test_pred - y_testo, c='orange', marker='*',s=10, label='Test dataset')
plt.scatter(y_pred, y_pred - yt, c='red', marker='.',s=7, label='Future prediction')
plt.xlabel('Predicted values [m]')
plt.ylabel('Residuals [m]')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-50, xmax=100, lw=2, color='k')
plt.xlim([-1, 3])
plt.ylim([-3, 2])
plt.show()