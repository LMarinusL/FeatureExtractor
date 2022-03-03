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
col_study = ['year', 'interval', 'x', 'y',  'hprevious', 'hrelative1','hrelative2','hrelative3', 'slope', 'aspect', 'curvature', 'dist', 'averageRunoff1', 'averageRunoff2', 'averageRunoff3', 'discharge','skeletonAngle']
param_study = 'hdifference'

###############

dfTrain = sklearn.utils.resample(df[df.year < 2017], n_samples=10000, random_state=None, stratify=None)
Xo = dfTrain[col_study]
yo = dfTrain[param_study]
dfTest = sklearn.utils.resample(df[df.year > 2017], n_samples=10000, random_state=None, stratify=None)
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

features=df.columns[[0,1,2,3,4,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17]]

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

#######################

df18 = sklearn.utils.resample(df[df.year == 2018], n_samples=10000, random_state=None, stratify=None)
df12 = sklearn.utils.resample(df[df.year == 2012], n_samples=10000, random_state=None, stratify=None)
df08 = sklearn.utils.resample(df[df.year == 2008], n_samples=10000, random_state=None, stratify=None)

print(df18.head())
fig4, ax4 = plt.subplots()
ax4.set_title('Annual sedimentation')
ax4.boxplot([df08['hdifference'],df12['hdifference'],df18['hdifference']], showfliers=False)
plt.xticks([1, 2, 3], ['97-08', '08-12', '12-18'])
plt.show()

###################

######################

col_study2 = ['interval',  'hprevious', 'hrelative1','hrelative2','hrelative3', 'slope', 'aspect', 'curvature', 'dist', 'averageRunoff1', 'averageRunoff2', 'averageRunoff3', 'discharge','skeletonAngle']
param_study = 'hdifference'

Xo2 = dfTrain[col_study2]
yo2 = dfTrain[param_study]
print('resampled')
Xt2 = dfTest[col_study2]
yt = dfTest[param_study]
X_traino2, X_testo2, y_traino2, y_testo2 = train_test_split(Xo2, yo2, test_size=0.3, random_state=42)
print('split')
forest3 = RandomForestRegressor()
forest3.fit(X_traino2, y_traino2)
print('fitted')
y_train_pred = forest3.predict(X_traino2)
y_test_pred = forest3.predict(X_testo2)
y_pred = forest3.predict(Xt2)
print('predicted')


###############################
plt.rcParams.update({'font.size': 15})

result = pd.DataFrame(forest3.feature_importances_,  df[col_study2].columns)
result.columns = ['importance']
result.sort_values(by='importance', ascending=False)

print(df.head())
features=df.columns[[1,4,6,7,8,9, 10, 11, 12, 13, 14, 15, 16,17]]

importances2 = forest3.feature_importances_
indices = np.argsort(importances2)
fig1 = plt.figure()
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances2[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()
###################################

plt.rcParams.update({'font.size': 20})
fig5, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
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
sc = ax[1].scatter(Xt['x'], Xt['y'],
           linewidths=1, alpha=.7,
            edgecolor='none',
           s = 20,
           c=(y_pred),
            cmap=cm, vmin=-2, vmax=2)
cbar = fig5.colorbar(sc)
cbar.ax.set_ylabel('Change in bed level height per year [m] without xy', rotation=270)
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

fig5.subplots_adjust(wspace=0.03, hspace=0)
fig5.suptitle('Annual sedimentation 2012-2018')
plt.show()
