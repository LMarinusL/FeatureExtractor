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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import colors
import graphviz
from sklearn.tree import export_graphviz



import numpy as np
plt.rcParams.update({'font.size': 16})

#https://epsg.io/transform#s_srs=4326&t_srs=32617&x=-79.6170883&y=9.1944523
#https://openaltimetry.org/data/icesat2/

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
print(df.columns)

##############

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

################
#0-year 1-interval 2-x 3-y 4-depth 5-hdifference 6-hrelative1 7-hrelative2 8-hrelative3 9-slope 10-aspect 11-curvatureS 12-curvatureM 13-curvatureL 14-averageRunoff1 15-averageRunoff2 16-averageRunoff3 17-discharge 18-skeletonAngleChagres 19-riverLengthChagres 20-inflowChagres 21-distChagres 22-skeletonAnglePequeni 23-riverLengthPequeni 24-inflowPequeni 25-distPequeni 26-profileCurvature 27-planformCurvature
col_study = ['year','interval','x','y','depth',  'hrelative1', 'hrelative2', 'hrelative3', 'slope', 'aspect', 'curvatureS','curvatureM','curvatureL', 'averageRunoff1','averageRunoff2', 'averageRunoff3','discharge','skeletonAngleChagres', 'riverLengthChagres', 'inflowChagres', 'distChagres', 'skeletonAnglePequeni', 'riverLengthPequeni', 'inflowPequeni', 'distPequeni', 'random', 'averageSlope', 'index', 'totalDistChagres','totalDistPequeni']
param_study = 'hdifference'

#[df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970]
"""dfTrain = sklearn.utils.resample(df[df.year < 2012][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
Xo = dfTrain[col_study]
yo = dfTrain[param_study]
dfTest = sklearn.utils.resample(df[df.year == 2012][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
"""

dfTrain = sklearn.utils.resample(df[df.year < 2012][df.hdifference > -2][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
Xo = dfTrain[col_study]
yo = dfTrain[param_study]
dfTest = sklearn.utils.resample(df[df.year == 2012][df.hdifference > -2][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)

Xt = dfTest[col_study]
yt = np.clip(dfTest[param_study], -5, 2)



print(yt.max())

#######################
# BOXPLOT OF DATA
#######################
df12T = sklearn.utils.resample(df[df.year == 2012][df.hdifference > -10][df.hdifference < 10], n_samples=10000, random_state=None, stratify=None)
df08T = sklearn.utils.resample(df[df.year == 2008][df.hdifference > -10][df.hdifference < 10], n_samples=10000, random_state=None, stratify=None)
df97T = sklearn.utils.resample(df[df.year == 1997][df.hdifference > -10][df.hdifference < 10], n_samples=10000, random_state=None, stratify=None)

df12H = sklearn.utils.resample(df[df.year == 2012][df.hdifference > -10][df.hdifference < 10][df.x < 1489], n_samples=10000, random_state=None, stratify=None)
df08H = sklearn.utils.resample(df[df.year == 2008][df.hdifference > -10][df.hdifference < 10][df.x < 1489], n_samples=10000, random_state=None, stratify=None)
df97H = sklearn.utils.resample(df[df.year == 1997][df.hdifference > -10][df.hdifference < 10][df.x < 1489], n_samples=10000, random_state=None, stratify=None)

df12P = sklearn.utils.resample(df[df.year == 2012][df.hdifference > -10][df.hdifference < 10][df.x < 1489][df.x > 1049][df.y < 650][df.y > 525], n_samples=10000, random_state=None, stratify=None)
df08P = sklearn.utils.resample(df[df.year == 2008][df.hdifference > -10][df.hdifference < 10][df.x < 1489][df.x > 1049][df.y < 650][df.y > 525], n_samples=10000, random_state=None, stratify=None)
df97P = sklearn.utils.resample(df[df.year == 1997][df.hdifference > -10][df.hdifference < 10][df.x < 1489][df.x > 1049][df.y < 650][df.y > 525], n_samples=10000, random_state=None, stratify=None)

"""df12 = sklearn.utils.resample(df[df.year == 2012][df.hdifference > -10][df.hdifference < 10][df.x < 1888][df.x > 1535], n_samples=10000, random_state=None, stratify=None)
df08 = sklearn.utils.resample(df[df.year == 2008][df.hdifference > -10][df.hdifference < 10][df.x < 1888][df.x > 1535], n_samples=10000, random_state=None, stratify=None)
df97 = sklearn.utils.resample(df[df.year == 1997][df.hdifference > -10][df.hdifference < 10][df.x < 1888][df.x > 1535], n_samples=10000, random_state=None, stratify=None)
"""

df12 = sklearn.utils.resample(df[df.year == 2012][df.hdifference > -2][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
df08 = sklearn.utils.resample(df[df.year == 2008][df.hdifference > -2][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
df97 = sklearn.utils.resample(df[df.year == 1997][df.hdifference > -2][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)



yt97 = df97[param_study] * 0.6
yt08 = df08[param_study]
yt12 = df12[param_study] * 0.8

print(" full volume 97-08 diff: "+str(yt97.sum()))
print(" full volume 08-12 diff: "+str(yt08.sum()))
print(" full volume 12-18 diff: "+str(yt12.sum()))

fig4, ax4 = plt.subplots()
ax4.set_title('Morphological changes for Chagres')
ax4.set_ylim(-3,4)
ax4.boxplot([df97['hdifference'],df08['hdifference'],df12['hdifference']], showfliers=False)
ax4.violinplot([df97T['hdifference'],df08T['hdifference'],df12T['hdifference']], showextrema=False)
plt.xticks([1, 2, 3], ['97-08', '08-12', '12-18'])
plt.draw()

fig4, ax4 = plt.subplots()
ax4.set_title('Morphological changes for Rio Pequeni')
ax4.set_ylim(-3,4)
ax4.boxplot([df97P['hdifference'],df08P['hdifference'],df12P['hdifference']], showfliers=False)
ax4.violinplot([df97P['hdifference'],df08P['hdifference'],df12P['hdifference']], showextrema=False)
plt.xticks([1, 2, 3], ['97-08', '08-12', '12-18'])
plt.draw()

fig4, ax4 = plt.subplots()
ax4.set_title('Alhajuela Lake total morphological changes')
ax4.set_ylim(-3,4)
ax4.boxplot([df97T['hdifference'],df08T['hdifference'],df12T['hdifference']], showfliers=False)
ax4.violinplot([df97T['hdifference'],df08T['hdifference'],df12T['hdifference']], showextrema=False)
plt.xticks([1, 2, 3], ['97-08', '08-12', '12-18'])
plt.draw()

fig4, ax4 = plt.subplots()
ax4.set_title('Alhajuela Lake main basin morphological changes')
ax4.set_ylim(-3,4)
ax4.boxplot([df97H['hdifference'],df08H['hdifference'],df12H['hdifference']], showfliers=False)
ax4.violinplot([df97H['hdifference'],df08H['hdifference'],df12H['hdifference']], showextrema=False)
plt.xticks([1, 2, 3], ['97-08', '08-12', '12-18'])
plt.draw()

print(df12.head())
def plotHeightForYears():


    plt.rcParams.update({'font.size': 20})
    fig5, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True,
                                        figsize=(20, 10))
    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax[0].scatter(df97['x'], df97['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df97['height']),
                cmap=cm, vmin=60, vmax=70)
    ax[0].set_title('97')
    ax[0].set_xlabel("x coordinate")
    ax[0].set_ylabel("y coordinate")
    ax[0].tick_params(labelsize=12)
    sc = ax[1].scatter(df08['x'], df08['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df08['height']),
                cmap=cm, vmin=60, vmax=70)
    cbar = fig5.colorbar(sc)
    cbar.ax.set_ylabel('bed level height [m]', rotation=270)
    cbar.ax.get_yaxis().labelpad = 20
    ax[1].set_title('08')
    ax[1].set_xlabel("x coordinate")
    ax[1].tick_params(labelsize=12)
    sc = ax[2].scatter(df12['x'], df12['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df12['height']),
                cmap=cm,  vmin=60, vmax=70)
    ax[2].set_title('12')
    ax[2].set_xlabel("x coordinate")
    ax[2].tick_params(labelsize=12)
    sc = ax[3].scatter(df12['x'], df12['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df12['height'] + df12['hdifference']),
                cmap=cm,  vmin=60, vmax=70)
    ax[3].set_title('18')
    ax[3].set_xlabel("x coordinate")
    ax[3].tick_params(labelsize=12)

    fig5.subplots_adjust(wspace=0.03, hspace=0.05)
    fig5.suptitle('Height over the yeears' )
    plt.draw()
plotHeightForYears()

print(df12.head())
def plotDiffForYears():

    plt.rcParams.update({'font.size': 10})
    fig52, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                                        figsize=(20, 10))
    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax[0].scatter(df97T['x'], df97T['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(df97T['hdifference']),
                cmap=cm,  vmin=-2, vmax=2)
    ax[0].set_title('97-08')
    ax[0].set_xlabel("x coordinate")
    ax[0].set_ylabel("y coordinate")
    ax[0].tick_params(labelsize=12)
    sc = ax[1].scatter(df08T['x'], df08T['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(df08T['hdifference']),
                cmap=cm, vmin=-2, vmax=2)
    cbar = fig52.colorbar(sc)
    cbar.ax.set_ylabel('difference in height [m]', rotation=270)
    cbar.ax.get_yaxis().labelpad = 20
    ax[1].set_title('08-12')
    ax[1].set_xlabel("x coordinate")
    ax[1].tick_params(labelsize=12)
    sc = ax[2].scatter(df12T['x'], df12T['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(df12T['hdifference']),
                cmap=cm,  vmin=-2, vmax=2)
    ax[2].set_title('12-18')
    ax[2].set_xlabel("x coordinate")
    ax[2].tick_params(labelsize=12)


    fig52.subplots_adjust(wspace=0.03, hspace=0.05)
    fig52.suptitle('Difference over the yeears' )
    plt.draw()
plotDiffForYears()



###################
# TRAINING ALGORITHMS
######################
#0-year 1-interval 2-x 3-y 4-depth 5-hdifference 6-hrelative1 7-hrelative2 8-hrelative3 9-slope 10-aspect 11-curvatureS 12-curvatureM 13-curvatureL 14-averageRunoff1 15-averageRunoff2 16-averageRunoff3 17-discharge 18-skeletonAngleChagres 19-riverLengthChagres 20-inflowChagres 21-distChagres 22-skeletonAnglePequeni 23-riverLengthPequeni 24-inflowPequeni 25-distPequeni 26-random 27-averageSlope 28-index 29-height 30-totalDistChagres
col_study2 = [ 'depth', 'hrelative3', 'slope', 'curvatureS','curvatureM','curvatureL', 'averageRunoff1', 'averageRunoff2','averageRunoff3','discharge','skeletonAngleChagres', 'riverLengthChagres', 'distChagres', 'random', 'averageSlope', 'totalDistChagres','skeletonAnglePequeni', 'riverLengthPequeni', 'inflowPequeni', 'distPequeni','totalDistPequeni']
#col_study2 = ['interval',  'hprevious', 'hrelative1','hrelative2','hrelative3', 'slope', 'aspect', 'curvature', 'dist', 'averageRunoff1', 'averageRunoff2', 'averageRunoff3', 'discharge','skeletonAngle', 'riverLength', 'inflow']
param_study = 'hdifference'

Xo2 = dfTrain[col_study2]
yo2 = dfTrain[param_study]
Xt2 = dfTest[col_study2]
X_traino2, X_testo2, y_traino2, y_testo2 = train_test_split(Xo2, yo2, test_size=0.3, random_state=42)

def predict(alg, Xtrain, ytrain, Xpredict):
    algorithm = make_pipeline(StandardScaler(), alg)
    algorithm.fit(Xtrain, ytrain)
    prediction = algorithm.predict(Xpredict)
    return prediction

forest3 = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 200, bootstrap= False)
y_pred = predict(forest3, X_traino2, y_traino2, Xt2)


def rmse(actual, pred):
    return np.sqrt(((pred - actual) ** 2).mean())

def predictOtherAlgs(actual, Xt, X_traino2, y_traino2, Xt2, title):
    print('SVR-predicting: ')
    regr = SVR(C=1.0, epsilon=0.2)
    y_predSVR = predict(regr, X_traino2, y_traino2, Xt2)
    RMSE_SVR = rmse(actual, y_predSVR)
    """
    print('Gaussian-predicting: ')
    kernel = DotProduct() + WhiteKernel()
    GPR = GaussianProcessRegressor(kernel=kernel, random_state=0)
    y_predGPR = predict(GPR, X_traino2, y_traino2, Xt2)
    RMSE_GPR = rmse(actual, y_predGPR)
    """

    print('MLPR-predicting: ')
    MLPR = MLPRegressor(alpha=1e-05, random_state=1, max_iter=500, learning_rate='adaptive', solver='sgd')
    y_predMLPR = predict(MLPR, X_traino2, y_traino2, Xt2)
    RMSE_MLPR = rmse(actual, y_predMLPR)

    forest3 = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 200, bootstrap= False)
    y_pred = predict(forest3, X_traino2, y_traino2, Xt2)
    RMSE_RFR = rmse(actual, y_pred)


    plt.rcParams.update({'font.size': 8})
    fig5, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True,
                                        figsize=(20, 10))
    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax[0].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(actual),
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
    ax[1].set_title('SVR  RMSE: '+ str(RMSE_SVR))
    ax[1].set_xlabel("x coordinate")
    ax[1].tick_params(labelsize=12)
    sc = ax[2].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(y_predMLPR),
                cmap=cm,  vmin=-2, vmax=2)
    ax[2].set_title('MLPR  RMSE: '+ str(RMSE_MLPR))
    ax[2].set_xlabel("x coordinate")
    ax[2].tick_params(labelsize=12)
    sc = ax[3].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(y_pred),
                cmap=cm,  vmin=-2, vmax=2)
    ax[3].set_title('RFR  RMSE: '+ str(RMSE_RFR))
    ax[3].set_xlabel("x coordinate")
    ax[3].tick_params(labelsize=12)

    fig5.subplots_adjust(wspace=0.03, hspace=0)
    fig5.suptitle(title)
    plt.draw()
    return
    
#predictOtherAlgs(yt, Xt, X_traino2, y_traino2, Xt2, 'all features')
###################################
# RF IMPORTANCES
###############################
forestImportance = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 50, bootstrap= False)
features=df.columns[[4,8,9, 11, 12, 13, 14,15,16,17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 30, 31]]

def plotImportances(imp, feat, Xtrain, ytrain):
    imp.fit(Xtrain, ytrain)
    plt.rcParams.update({'font.size': 8})

    result = pd.DataFrame(imp.feature_importances_,  df[feat].columns)
    result.columns = ['importance']
    result.sort_values(by='importance', ascending=False)

    importances2 = imp.feature_importances_
    indices = np.argsort(importances2)
    fig1 = plt.figure()
    plt.title('Feature Importances - number of features:' + str(feat.size))
    plt.barh(range(len(indices)), importances2[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feat[indices])
    plt.xlabel('Relative Importance')
    plt.draw()
    return indices

indices = plotImportances(forestImportance, features, X_traino2, y_traino2)


###############################
# SET WITH LESS FEATURES
##################################

col_study3 = features[indices[7:-1]]
Xo3 = dfTrain[col_study3]
yo3 = dfTrain[param_study]
Xt3 = dfTest[col_study3]
X_traino2, X_testo2, y_traino2, y_testo2 = train_test_split(Xo3, yo3, test_size=0.3, random_state=42)
forest5 = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 50, bootstrap= False)
print('rf-predicting less features: ')
y_predXS = predict(forest5, X_traino2, y_traino2, Xt3)

#predictOtherAlgs(yt, Xt, X_traino2, y_traino2, Xt3, 'Only highest impurity features')
############################################
# tests with range of features
###########################################
#0-year 1-interval 2-x 3-y 4-depth 5-hdifference 6-hrelative1 7-hrelative2 8-hrelative3 9-slope 10-aspect 11-curvatureS 12-curvatureM 13-curvatureL 14-averageRunoff1 15-averageRunoff2 16-averageRunoff3 17-discharge 18-skeletonAngleChagres 19-riverLengthChagres 20-inflowChagres 21-distChagres 22-skeletonAnglePequeni 23-riverLengthPequeni 24-inflowPequeni 25-distPequeni 26-random 27-averageSlope 28-index 29-height 30-totalDistChagres
def plotSelection(col_study, yt, Xt, dfTrain, dfTest, title):
    XoP = dfTrain[col_study]
    yoP = dfTrain[param_study]
    XtP = dfTest[col_study]
    predictOtherAlgs(yt, Xt, XoP, yoP, XtP, title)

col_study_All_Chagres = [ 'depth',  'hrelative1', 'hrelative2', 'hrelative3', 'slope', 'aspect', 'curvatureS','curvatureM','curvatureL', 'averageRunoff1','averageRunoff2', 'averageRunoff3','discharge','skeletonAngleChagres', 'riverLengthChagres', 'inflowChagres', 'distChagres', 'averageSlope', 'totalDistChagres']
col_study_No_Curvature = [ 'depth',  'hrelative1', 'hrelative2', 'hrelative3', 'slope', 'aspect', 'averageRunoff1','averageRunoff2', 'averageRunoff3','discharge','skeletonAngleChagres', 'riverLengthChagres', 'inflowChagres', 'distChagres', 'averageSlope', 'totalDistChagres']
col_study_No_hrelative = [ 'depth', 'slope', 'aspect', 'curvatureS','curvatureM','curvatureL', 'averageRunoff1','averageRunoff2', 'averageRunoff3','discharge','skeletonAngleChagres', 'riverLengthChagres', 'inflowChagres', 'distChagres', 'averageSlope', 'totalDistChagres']
col_study_No_skeleton = [ 'depth',  'hrelative1', 'hrelative2', 'hrelative3', 'slope', 'aspect', 'curvatureS','curvatureM','curvatureL', 'averageRunoff1','averageRunoff2', 'averageRunoff3','discharge', 'inflowChagres', 'averageSlope']
col_study_No_runoff = [ 'depth',  'hrelative1', 'hrelative2', 'hrelative3', 'slope', 'aspect', 'curvatureS','curvatureM','curvatureL', 'discharge','skeletonAngleChagres', 'riverLengthChagres', 'inflowChagres', 'distChagres', 'averageSlope', 'totalDistChagres']
col_study_handpicked = [  'depth', 'aspect','curvatureM','averageRunoff2', 'averageRunoff3','riverLengthPequeni', 'distPequeni', 'averageSlope','totalDistPequeni']
forestImportanceP = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 50, bootstrap= False)
featuresP=df.columns[[4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,27,30]]


"""indicesP = plotImportances(forestImportanceP, featuresP, dfTrain[featuresP], dfTrain[param_study] )
print(featuresP[indicesP])
title = 'params number: '+ str(indicesP.size)
plotSelection(featuresP[indicesP], yt, Xt, dfTrain, dfTest, title)

for i in range(1):
    featuresP = featuresP[indicesP[1:]]
    forestImportanceP = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 50, bootstrap= False)
    indicesP = plotImportances(forestImportanceP, featuresP, dfTrain[featuresP], dfTrain[param_study] )
    print(featuresP[indicesP])
    title = 'params number: '+ str(indicesP.size)
    plotSelection(featuresP[indicesP], yt, Xt, dfTrain, dfTest, title)
"""
"""dfTestNew10 = dfTest.assign(inflowChagres=10)
dfTestNew20 = dfTest.assign(inflowChagres=20)
dfTestNew30 = dfTest.assign(inflowChagres=30)
dfTestNew40 = dfTest.assign(inflowChagres=40)
dfTestNew50 = dfTest.assign(inflowChagres=50)
"""
#plotSelection(col_study_handpicked, yt, Xt, dfTrain, dfTest, 'handpicked')



##################################
# CV
########################################

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestRegressor()

#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#rf_random.fit(X_traino2, y_traino2)
#print(rf_random.best_params_)
"""
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions  - test_labels)
    mape = 100 * np.mean(errors / (test_labels+ test_features['hprevious']))
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} meters.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators = 10, random_state = 42))
base_model.fit(X_traino2, y_traino2)
Xt6 = dfTest[col_study2]
yt6 = dfTest[param_study]
base_accuracy = evaluate(base_model, Xt6, yt6)
optimized_accuracy = evaluate(forest3, Xt6, yt6)
print('Improvement of {:0.2f}%.'.format( 100 * (optimized_accuracy - base_accuracy) / base_accuracy))
"""


###################################
# RF BOXPLOT
###################################
def boxplot(actual, prediction, title):
    figbox = plt.figure()
    plt.title(title)
    plt.boxplot([prediction, actual, (prediction - actual)], showfliers=False)
    plt.xticks([1, 2, 3], ['Prediction', 'Actual', 'Residuals'])
    plt.ylabel('[m]')
    plt.draw()
    return figbox

boxplot(yt, y_pred, '2012-2018 Sedimentation RF')
boxplot(yt, y_predXS, '2012-2018 Sedimentation RF less features')

"""
fig3 = plt.figure()
plt.title('2012-2018 Sedimentation RF')
plt.boxplot([y_pred, yt, (y_pred - yt)], showfliers=False)
plt.xticks([1, 2, 3], ['Prediction', 'Actual', 'Residuals'])
plt.ylabel('[m]')
plt.draw()

fig3_s = plt.figure()
plt.title('2012-2018 Sedimentation RF less features')
plt.boxplot([y_predXS, yt, (y_predXS - yt)], showfliers=False)
plt.xticks([1, 2, 3], ['Prediction', 'Actual', 'Residuals'])
plt.ylabel('[m]')
plt.draw()
"""


print(" all params MAE: ", mean_absolute_error(yt, y_pred))
print(" only important params MAE: ", mean_absolute_error(yt, y_predXS))

###################################
# RF PLOT
###################################

def plotMaps2Sets(actual, pred1, pred2):
    plt.rcParams.update({'font.size': 8})
    fig5, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                                        figsize=(20, 10))
    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax[0,0].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(actual),
                cmap=cm, vmin=-2, vmax=2)
    ax[0,0].set_title('Actual')
    ax[0,0].set_xlabel("x coordinate")
    ax[0,0].set_ylabel("y coordinate")
    ax[0,0].tick_params(labelsize=12)
    sc = ax[0,1].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(pred1),
                cmap=cm, vmin=-2, vmax=2)
    cbar = fig5.colorbar(sc)
    cbar.ax.set_ylabel('Change in bed level height per year [m] without xy', rotation=270)
    cbar.ax.get_yaxis().labelpad = 40
    ax[0,1].set_title('Prediction')
    ax[0,1].set_xlabel("x coordinate")
    ax[0,1].tick_params(labelsize=12)
    sc = ax[0,2].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(pred1 - actual),
                cmap=cm,  vmin=-2, vmax=2)
    ax[0,2].set_title('Residual')
    ax[0,2].set_xlabel("x coordinate")
    ax[0,2].tick_params(labelsize=12)

    sc = ax[1,0].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(actual),
                cmap=cm, vmin=-2, vmax=2)
    ax[1,0].set_title('Actual')
    ax[1,0].set_xlabel("x coordinate")
    ax[1,0].set_ylabel("y coordinate")
    ax[1,0].tick_params(labelsize=12)
    sc = ax[1,1].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(pred2),
                cmap=cm, vmin=-2, vmax=2)
    cbar.ax.get_yaxis().labelpad = 20
    ax[1,1].set_title('Prediction with only high importance params')
    ax[1,1].set_xlabel("x coordinate")
    ax[1,1].tick_params(labelsize=12)
    sc = ax[1,2].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(pred2 - actual),
                cmap=cm,  vmin=-2, vmax=2)
    ax[1,2].set_title('Residual')
    ax[1,2].set_xlabel("x coordinate")
    ax[1,2].tick_params(labelsize=12)

    fig5.subplots_adjust(wspace=0.03, hspace=0)
    fig5.suptitle('RF Annual sedimentation 2012-2018')
    plt.draw()
    return

plotMaps2Sets(yt, y_pred, y_predXS)


###################################
# SVR PLOT
###################################
def plotMaps1Set(actual, pred, title):
    plt.rcParams.update({'font.size': 8})
    fig5, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                                        figsize=(20, 10))
    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax[0].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(actual),
                cmap=cm, vmin=-2, vmax=2)
    ax[0].set_title('Actual')
    ax[0].set_xlabel("x coordinate")
    ax[0].set_ylabel("y coordinate")
    ax[0].tick_params(labelsize=12)
    sc = ax[1].scatter(Xt['x'], Xt['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(pred),
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
            c=(pred - actual),
                cmap=cm,  vmin=-2, vmax=2)
    ax[2].set_title('Residual')
    ax[2].set_xlabel("x coordinate")
    ax[2].tick_params(labelsize=12)

    fig5.subplots_adjust(wspace=0.03, hspace=0)
    fig5.suptitle(title)
    plt.draw()
    return

def plotMaps1Binned(actual, pred, title):
    x_array = Xt['x'].to_numpy()
    y_array = Xt['y'].to_numpy()
    actual_array = actual.to_numpy()
    ranges = [-5, -1.5, -0.5, 0.5, 1.5, 20]

    plt.rcParams.update({'font.size': 8})
    fig5, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    for i in range(pred.size):
        if ranges[0] < pred[i] < ranges[1]:
            sc = axs[0].scatter(x_array[i], y_array[i], c='red')
        if ranges[1] < pred[i] < ranges[2]:
            sc = axs[0].scatter(x_array[i], y_array[i], c='orange')
        if ranges[2] < pred[i] < ranges[3]:
            sc = axs[0].scatter(x_array[i], y_array[i], c='yellow')
        if ranges[3] < pred[i] < ranges[4]:
            sc = axs[0].scatter(x_array[i], y_array[i], c='green')
        if ranges[4] < pred[i] < ranges[5]:
            sc = axs[0].scatter(x_array[i], y_array[i], c='blue')

    for i in range(actual_array.size):
        if ranges[0] < actual_array[i] < ranges[1]:
            sc = axs[1].scatter(x_array[i], y_array[i], c='red')
        if ranges[1] < actual_array[i] < ranges[2]:
            sc = axs[1].scatter(x_array[i], y_array[i], c='orange')
        if ranges[2] < actual_array[i] < ranges[3]:
            sc = axs[1].scatter(x_array[i], y_array[i], c='yellow')
        if ranges[3] < actual_array[i] < ranges[4]:
            sc = axs[1].scatter(x_array[i], y_array[i], c='green')
        if ranges[4] < actual_array[i] < ranges[5]:
            sc = axs[1].scatter(x_array[i], y_array[i], c='blue')

    axs[0].set_title('Prediction')
    axs[1].set_title('Actual')

    fig5.subplots_adjust(wspace=0.03, hspace=0)
    fig5.suptitle(title)
    plt.draw()
    return



#plotMaps1Binned(yt, y_pred, 'Binned')


###################################
# GPR PLOT
###################################

#plotMaps1Set(yt, y_predGPR, 'GPR Annual sedimentation 2012-2018')


###################################
# MLPR PLOT
###################################

#plotMaps1Set(yt, y_predMLPR, 'MLPR Annual sedimentation 2012-2018')

################
# HISTOGRAM
####################
def plotErrorHist(actual, pred):

    plt.rcParams.update({'font.size': 8})

    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=False, tight_layout=True)
    hist1 = ax[0,0].hist2d(actual, 
           (pred - actual),
           bins = [80,80], 
           cmap = "RdYlGn_r",
           norm = colors.LogNorm(),
           density = True)     
    ax[0,0].set_title('error on actual')
    ax[0,0].set_xlabel("actual")
    ax[0,0].set_ylabel("error")
    #ax[0].set_xlim((-1,4))

    hist2 = ax[0,1].hist2d(pred, 
           (pred - actual),
           bins = [80,80], 
           cmap = "RdYlGn_r",
           norm = colors.LogNorm(),
           density = True)
    ax[0,1].set_title('error on prediction')
    fig.colorbar(hist2[3],ax=ax[0,1])    
    ax[0,1].set_xlabel("prediction")
    ax[0,1].set_ylabel("error")

    number = 0
    hist1Temp =  hist1[1][1:10]
    for line in hist1Temp:
        histTemp =  hist1[2][1:]
        ax[1,0].plot(histTemp, hist1[0][number], label=hist1Temp[number])
        number = number + 1
    ax[1,0].set_xlim([-2, 2])
    number = 0
    hist2Temp =  hist2[1][1:10]
    for line in hist2Temp:
        histTemp =  hist2[2][1:]
        ax[1,1].plot(histTemp, hist2[0][number], label=hist2Temp[number])
        number = number + 1
    ax[1,1].legend(loc="upper right", prop={'size': 6})
    ax[1,1].set_xlim([-2, 2])
    plt.draw()
    return

"""plotErrorHist(yt, y_pred)
print(y_pred[89])
np.random.shuffle(y_pred)
print(y_pred[89])
plotErrorHist(yt, y_pred)"""



####################
# Writer
####################

forest6 = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 200, bootstrap= False)
svrModel = SVR(C=1.0, epsilon=0.2)
col_study_peq = [ 'depth', 'hrelative3','hrelative2', 'slope', 'curvatureL', 'averageRunoff1', 'averageRunoff2','averageSlope','riverLengthPequeni', 'distPequeni','totalDistPequeni']
col_study_chag = [ 'depth', 'riverLengthChagres','totalDistChagres', 'distChagres','averageRunoff3', 'averageRunoff2', 'averageRunoff1','aspect', 'averageSlope','skeletonAngleChagres','hrelative3','hrelative2','slope', 'curvatureL']

def predict(alg, Xtrain, ytrain, Xpredict):
    algorithm = make_pipeline(StandardScaler(), alg)
    algorithm.fit(Xtrain, ytrain)
    prediction = algorithm.predict(Xpredict)
    return prediction

dfpred = pd.read_csv('C:/Users/neder/Documents/Geomatics/Unity/PCproject/DEMViewer/Assets/Output/outputGridPredParams.txt', sep=" ")
X_arr = dfpred[col_study_chag][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970].to_numpy()
print("pred file read")
dfTrainFull = sklearn.utils.resample(df[df.year < 2013][df.hdifference > -2][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
set_X_T = dfTrainFull[col_study_chag]
set_y_T = dfTrainFull['hdifference']

pred = predict(svrModel, set_X_T, set_y_T, X_arr)
index_array = dfpred['index'][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970].to_numpy()
x_array = dfpred['x'][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970].to_numpy()
y_array = dfpred['y'][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970].to_numpy()
d_array = dfpred['height'][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970].to_numpy()
print("pred made")


outputFile =  open('C:/Users/neder/Documents/Geomatics/Unity/PCproject/DEMViewer/Assets/Output/Python_Output.txt', 'w')
for i in range(pred.size):
    outputFile.write(str(index_array[i]))
    outputFile.write( " ")
    outputFile.write(str(x_array[i]))
    outputFile.write( " ")
    outputFile.write(str(y_array[i]))
    outputFile.write(" ")
    outputFile.write(str(pred[i] + d_array[i] ))
    outputFile.write('\n')
outputFile.close()
print("output written")
#############################################################
# Hierarchy diagram
######################################
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs, orientation='left', show_leaf_counts=True, labels=model.labels_)
"""
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X_traino2, y_traino2)
fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True)
ax.set_title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram

plot_dendrogram(model, truncate_mode="level", p=3)
ax.set_xlabel("Number of points in node (or index of point if no parenthesis).")
plt.draw()
"""

##################
# GRAPH TREE 

dot_data = export_graphviz(forest3.estimators_[0], 
                        out_file='C:/Users/neder/Documents/Geomatics/Unity/PCproject/DEMViewer/Assets/Output/tree.dot',
                        feature_names=col_study2,
                        max_depth=5,
                           filled=True, impurity=True, 
                           rounded=True)

graph = graphviz.Source(dot_data, format='png')


#################################

def plotOnYears(property, min, max):
    """    df97 = sklearn.utils.resample(df[df.year == 1997][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df18 = sklearn.utils.resample(df[df.year == 2018][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df12 = sklearn.utils.resample(df[df.year == 2012][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df08 = sklearn.utils.resample(df[df.year == 2008][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df22 = sklearn.utils.resample(df[df.year == 2022][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    #df26 = sklearn.utils.resample(df[df.year == 2026][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    #df30 = sklearn.utils.resample(df[df.year == 2030][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    #df34 = sklearn.utils.resample(df[df.year == 2034][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    """

    df97 = sklearn.utils.resample(df[df.year == 1997][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df08 = sklearn.utils.resample(df[df.year == 2008][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df12 = sklearn.utils.resample(df[df.year == 2012][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df18 = sklearn.utils.resample(df[df.year == 2018][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df22 = sklearn.utils.resample(df[df.year == 2022][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df26 = sklearn.utils.resample(df[df.year == 2026][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    df30 = sklearn.utils.resample(df[df.year == 2030][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
    #df34 = sklearn.utils.resample(df[df.year == 2034][df.hdifference > -10][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)




    plt.rcParams.update({'font.size': 20})
    fig5, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True,
                                        figsize=(20, 10))
    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax[0,0].scatter(df97['x'], df97['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df97[property]),
                cmap=cm,  vmin=min, vmax=max)
    ax[0,0].set_title('97')
    ax[0,0].set_xlabel("x coordinate")
    ax[0,0].set_ylabel("y coordinate")
    ax[0,0].tick_params(labelsize=12)
    sc = ax[0,1].scatter(df08['x'], df08['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df08[property]),
                cmap=cm,  vmin=min, vmax=max)
    cbar = fig5.colorbar(sc)
    cbar.ax.get_yaxis().labelpad = 20
    ax[0,1].set_title('08')
    ax[0,1].set_xlabel("x coordinate")
    ax[0,1].tick_params(labelsize=12)
    sc = ax[0,2].scatter(df12['x'], df12['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df12[property]),
                cmap=cm,  vmin=min, vmax=max)
    ax[0,2].set_title('12')
    ax[0,2].set_xlabel("x coordinate")
    ax[0,2].tick_params(labelsize=12)
    
    sc = ax[0,3].scatter(df18['x'], df18['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df18[property]),
                cmap=cm, vmin=min, vmax=max)
    ax[0,3].set_title('18')
    ax[0,3].set_xlabel("x coordinate")
    ax[0,3].tick_params(labelsize=12)
    
    sc = ax[1,0].scatter(df22['x'], df22['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df22[property]),
                cmap=cm, vmin=min, vmax=max)
    ax[1,0].set_title('22')
    ax[1,0].set_xlabel("x coordinate")
    ax[1,0].set_ylabel("y coordinate")
    ax[1,0].tick_params(labelsize=12)
    
    sc = ax[1,1].scatter(df26['x'], df26['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df26[property]),
                cmap=cm, vmin=min, vmax=max)

    ax[1,1].set_title('26')
    ax[1,1].set_xlabel("x coordinate")
    ax[1,1].tick_params(labelsize=12)
    
    sc = ax[1,2].scatter(df30['x'], df30['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df30[property]),
                cmap=cm, vmin=min, vmax=max)
    ax[1,2].set_title('30')
    ax[1,2].set_xlabel("x coordinate")
    ax[1,2].tick_params(labelsize=12)
    """
    sc = ax[1,3].scatter(df34['x'], df34['y'],
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(df34[property]),
                cmap=cm, vmin=min, vmax=max)
    ax[1,3].set_title('34')
    ax[1,3].set_xlabel("x coordinate")
    ax[1,3].tick_params(labelsize=12)
    
    """
    fig5.subplots_adjust(wspace=0.03, hspace=0.05)
    fig5.suptitle(property)
    plt.draw()

plotOnYears('hdifference', -2, 2)
plotOnYears('height', 50, 70)
plotOnYears('distChagres', 0, 40)


##################################
# PROB
########################################

def probSVR(iter, limit, col_study, dfTrain, dfTest):
    XoTrain = dfTrain[col_study]
    yoTrain = dfTrain[param_study]
    XtTest = dfTest[col_study]

    print('computing probabilities')
    values = [0] *dfTest['y'].size
    for i in range(iter):
        algNew = SVR(C=1.0, epsilon=0.2)
        y_pred = predict(algNew, XoTrain, yoTrain, XtTest)
        for i in range(y_pred.size):
            if y_pred[i] > limit:
                values[i] = values[i] + 1
    return values

def probRFR(iter, limit, col_study, dfTrain, dfTest):
    XoTrain = dfTrain[col_study]
    yoTrain = dfTrain[param_study]
    XtTest = dfTest[col_study]

    print('computing probabilities')
    values = [0] *dfTest['y'].size
    for i in range(iter):
        algNew = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 200, bootstrap= False)
        y_pred = predict(algNew, XoTrain, yoTrain, XtTest)
        for i in range(y_pred.size):
            if y_pred[i] > limit:
                values[i] = values[i] + 1
    return values

def probMLPR(iter, limit, col_study, dfTrain, dfTest):
    XoTrain = dfTrain[col_study]
    yoTrain = dfTrain[param_study]
    XtTest = dfTest[col_study]

    print('computing probabilities')
    values = [0] *dfTest['y'].size
    for i in range(iter):
        algNew = MLPRegressor(alpha=1e-05, random_state=1, max_iter=500, learning_rate='adaptive', solver='sgd')
        y_pred = predict(algNew, XoTrain, yoTrain, XtTest)
        for i in range(y_pred.size):
            if y_pred[i] > limit:
                values[i] = values[i] + 1
    return values


def plotProb(limit, values, dfTest, title):        
    X_arr = dfTest['x']
    Y_arr = dfTest['y']
    Diff_arr = dfTest['hdifference'].to_numpy()
    ActualValues = [0] *dfTest['y'].size
    margin = 0.4
    for i in range(Diff_arr.size):
        if Diff_arr[i] > (limit + margin):
            ActualValues[i] = ActualValues[i] + 1
    total = 0
    for i in range(len(values)):
        if values[i] == 0 & ActualValues[i] == 0:
            total = total + 1
        if values[i] != 0 & ActualValues[i] != 0:
            total = total + 1
    accuracy = total/len(values)
    plt.rcParams.update({'font.size': 8})
    fig5, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                                        figsize=(20, 10))
    cm = plt.cm.get_cmap('binary')
    sc = ax[0].scatter(X_arr, Y_arr,
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(values),
                cmap=cm, vmin=-2, vmax=2)
    subtitle = 'Probability of sedimentation above: ' + str(limit) + 'accuracy :' + str(accuracy)
    ax[0].set_title(subtitle)
    ax[0].set_xlabel("x coordinate")
    ax[0].set_ylabel("y coordinate")
    ax[0].tick_params(labelsize=12)    
    sc = ax[1].scatter(X_arr, Y_arr,
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 10,
            c=(ActualValues),
                cmap=cm, vmin=-2, vmax=2)
    ax[1].set_title('Actual')
    ax[1].set_xlabel("x coordinate")
    ax[1].set_ylabel("y coordinate")
    ax[1].tick_params(labelsize=12)
   
    fig5.subplots_adjust(wspace=0.03, hspace=0)
    fig5.suptitle(title)
    plt.draw()
    return


valuesRFR = probRFR(5, 0.5, col_study_handpicked, dfTrain, dfTest)
plotProb(0.5, valuesRFR, dfTest, 'RFR')

valuesSVR = probSVR(5, 0.5, col_study_handpicked, dfTrain, dfTest)
plotProb(0.5, valuesSVR, dfTest, 'SVR')

valuesSVR = probMLPR(5, 0.5, col_study_handpicked, dfTrain, dfTest)
plotProb(0.5, valuesSVR, dfTest, 'MPLR')


###################################
# CUMULATIVE ERROR
#####################################

col_studyPe = [ 'depth', 'hrelative3', 'slope', 'curvatureS','curvatureM','curvatureL', 'averageRunoff1', 'averageRunoff2','averageRunoff3','discharge','skeletonAngleChagres', 'riverLengthChagres', 'distChagres', 'random', 'averageSlope', 'totalDistChagres','skeletonAnglePequeni', 'riverLengthPequeni', 'inflowPequeni', 'distPequeni','totalDistPequeni']
#col_study2 = ['interval',  'hprevious', 'hrelative1','hrelative2','hrelative3', 'slope', 'aspect', 'curvature', 'dist', 'averageRunoff1', 'averageRunoff2', 'averageRunoff3', 'discharge','skeletonAngle', 'riverLength', 'inflow']
param_study = 'hdifference'

XoPe = dfTrain[col_study2]
yoPe = dfTrain[param_study]
XtPe = dfTest[col_study2]
X_trainoPe, X_testoPe, y_trainoPe, y_testoPe = train_test_split(Xo2, yo2, test_size=0.3, random_state=42)
svrPe = SVR(C=1.0, epsilon=0.2)
y_predPe = predict(svrPe, X_trainoPe, y_trainoPe, XtPe)

def plotCumulative(pred, actual, bins, title):
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, tight_layout=True)
    hist1 = ax[0].hist(
           (pred - actual),
           bins = bins,
           cumulative=True,
           histtype='stepfilled')     
    ax[0].set_xlabel("error")
    ax[0].set_ylabel("distribution")
    hist2 = ax[1].hist(
           abs(pred- actual),
           bins = bins, 
           cumulative=True,
           histtype='stepfilled')     
    ax[1].set_xlabel("absolute error")
    ax[1].set_ylabel("distribution")
    fig.suptitle(title)
plotCumulative(y_predPe, yt, 30, 'Cumulative Error Distribution')
print(max(y_predPe))
plotErrorHist(yt, y_predPe)

def plotActPred(actual, pred):

    plt.rcParams.update({'font.size': 8})

    fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True)
    hist1 = ax.hist2d(actual,
           pred,
           bins = [80,80], 
           cmap = "RdYlGn_r",
           norm = colors.LogNorm(),
           density = True)     
    ax.set_title('prediction and actual values')
    ax.set_xlabel("actual")
    ax.set_ylabel("pred")
plotActPred(yt, y_predPe)
plt.matshow(df.corr())
#################################
plt.show()