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
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


import numpy as np
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
col_study = ['year', 'interval', 'x', 'y',  'hprevious', 'hrelative1','hrelative2','hrelative3', 'slope', 'aspect', 'curvature', 'dist', 'averageRunoff1', 'averageRunoff2', 'averageRunoff3', 'discharge','skeletonAngle', 'riverLength', 'inflow']
param_study = 'hdifference'

###############

dfTrain = sklearn.utils.resample(df[df.year < 2017], n_samples=10000, random_state=None, stratify=None)
Xo = dfTrain[col_study]
yo = dfTrain[param_study]
dfTest = sklearn.utils.resample(df[df.year > 2017], n_samples=10000, random_state=None, stratify=None)
print('resampled')
Xt = dfTest[col_study]
yt = dfTest[param_study]


###################
# BOXPLOT OF DATA
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
# TRAINING ALGORITHMS
######################

col_study2 = ['interval',  'hprevious', 'hrelative1','hrelative2','hrelative3', 'slope', 'aspect', 'curvature', 'dist', 'averageRunoff1', 'averageRunoff2', 'averageRunoff3', 'discharge','skeletonAngle', 'riverLength', 'inflow']
param_study = 'hdifference'

Xo2 = dfTrain[col_study2]
yo2 = dfTrain[param_study]
print('resampled')
Xt2 = dfTest[col_study2]
yt = dfTest[param_study]
X_traino2, X_testo2, y_traino2, y_testo2 = train_test_split(Xo2, yo2, test_size=0.3, random_state=42)
print('split')
forestImportance = RandomForestRegressor(n_estimators=200)
forestImportance.fit(X_traino2, y_traino2)

forest3 = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200))

forest3.fit(X_traino2, y_traino2)
print('fitted')
print('rf-predicting: ')
y_train_pred = forest3.predict(X_traino2)
y_test_pred = forest3.predict(X_testo2)
y_pred = forest3.predict(Xt2)
print('predicted')

print('SVR-predicting: ')
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_traino2, y_traino2)
y_predSVR = regr.predict(Xt2)
print('predicted')

print('Gaussian-predicting: ')
kernel = DotProduct() + WhiteKernel()
GPR = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel,
        random_state=0))
GPR.fit(X_traino2, y_traino2)
y_predGPR = GPR.predict(Xt2)
print('predicted')

print('MLPR-predicting: ')
kernel = DotProduct() + WhiteKernel()
MLPR = make_pipeline(StandardScaler(), MLPRegressor(alpha=1e-05, random_state=1, max_iter=500, learning_rate='adaptive'))
MLPR.fit(X_traino2, y_traino2)
y_predMLPR = MLPR.predict(Xt2)
print('predicted')

###################################
# RF IMPORTANCES
###############################
plt.rcParams.update({'font.size': 15})

result = pd.DataFrame(forestImportance.feature_importances_,  df[col_study2].columns)
result.columns = ['importance']
result.sort_values(by='importance', ascending=False)

print(df.head())
features=df.columns[[1,4,6,7,8,9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19]]

importances2 = forestImportance.feature_importances_
indices = np.argsort(importances2)
fig1 = plt.figure()
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances2[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()
###################################
# RF PLOT
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
fig5.suptitle('RF Annual sedimentation 2012-2018')
plt.show()

###################################
# SVR PLOT
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
           c=(y_predSVR),
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
           c=(y_predSVR - yt),
            cmap=cm,  vmin=-2, vmax=2)
ax[2].set_title('Residual')
ax[2].set_xlabel("x coordinate")
ax[2].tick_params(labelsize=12)

fig5.subplots_adjust(wspace=0.03, hspace=0)
fig5.suptitle('SVR Annual sedimentation 2012-2018')
plt.show()

###################################
# GPR PLOT
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
           c=(y_predGPR),
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
           c=(y_predGPR - yt),
            cmap=cm,  vmin=-2, vmax=2)
ax[2].set_title('Residual')
ax[2].set_xlabel("x coordinate")
ax[2].tick_params(labelsize=12)

fig5.subplots_adjust(wspace=0.03, hspace=0)
fig5.suptitle('GPR Annual sedimentation 2012-2018')
plt.show()

###################################
# MLPR PLOT
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
           c=(y_predMLPR),
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
           c=(y_predMLPR - yt),
            cmap=cm,  vmin=-2, vmax=2)
ax[2].set_title('Residual')
ax[2].set_xlabel("x coordinate")
ax[2].tick_params(labelsize=12)

fig5.subplots_adjust(wspace=0.03, hspace=0)
fig5.suptitle('MLPR Annual sedimentation 2012-2018')
plt.show()