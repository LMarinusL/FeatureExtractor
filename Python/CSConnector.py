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
##############

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

################
#0-year 1-interval 2-x 3-y 4-hprevious 5-hdifference 6-hrelative1 7-hrelative2 8-hrelative3 9-slope 10-aspect 11-curvatureS 12-curvatureM 13-curvatureL 14-averageRunoff1 15-averageRunoff2 16-averageRunoff3 17-discharge 18-skeletonAngleChagres 19-riverLengthChagres 20-inflowChagres 21-distChagres 22-skeletonAnglePequeni 23-riverLengthPequeni 24-inflowPequeni 25-distPequeni 26-profileCurvature 27-planformCurvature
col_study = ['year','interval','x','y', 'hprevious',  'hrelative1', 'hrelative2', 'hrelative3', 'slope', 'aspect', 'curvatureS','curvatureM','curvatureL', 'averageRunoff1','averageRunoff2', 'averageRunoff3','discharge','skeletonAngleChagres', 'riverLengthChagres', 'inflowChagres', 'distChagres', 'skeletonAnglePequeni', 'riverLengthPequeni', 'inflowPequeni', 'distPequeni', 'random']
param_study = 'hdifference'

###############
#[df.y < -(7/2)*df.x + 3550][df.y > -1.25*df.x + 1575][df.x < 800][df.y < 900]
dfTrain = sklearn.utils.resample(df[df.year < 2017][df.hdifference > -1], n_samples=10000, random_state=None, stratify=None)
Xo = dfTrain[col_study]
yo = dfTrain[param_study]
dfTest = sklearn.utils.resample(df[df.year > 2017][df.hdifference > -1], n_samples=10000, random_state=None, stratify=None)
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

plt.rcParams.update({'font.size': 20})
fig5, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                                    figsize=(20, 10))
cm = plt.cm.get_cmap('RdYlBu')
sc = ax[0].scatter(df08['x'], df08['y'],
        linewidths=1, alpha=.7,
            edgecolor='none',
        s = 20,
        c=(df08['hdifference']),
            cmap=cm, vmin=-2, vmax=2)
ax[0].set_title('08')
ax[0].set_xlabel("x coordinate")
ax[0].set_ylabel("y coordinate")
ax[0].tick_params(labelsize=12)
sc = ax[1].scatter(df12['x'], df12['y'],
        linewidths=1, alpha=.7,
            edgecolor='none',
        s = 20,
        c=(df12['hdifference']),
            cmap=cm, vmin=-2, vmax=2)
cbar = fig5.colorbar(sc)
cbar.ax.set_ylabel('Change in bed level height per year [m] without xy', rotation=270)
cbar.ax.get_yaxis().labelpad = 20
ax[1].set_title('12')
ax[1].set_xlabel("x coordinate")
ax[1].tick_params(labelsize=12)
sc = ax[2].scatter(df18['x'], df18['y'],
        linewidths=1, alpha=.7,
            edgecolor='none',
        s = 20,
        c=(df18['hdifference']),
            cmap=cm,  vmin=-2, vmax=2)
ax[2].set_title('18')
ax[2].set_xlabel("x coordinate")
ax[2].tick_params(labelsize=12)

fig5.subplots_adjust(wspace=0.03, hspace=0)
fig5.suptitle('08, 12, 18 levels' )
plt.show()


###################
# TRAINING ALGORITHMS
######################
#0-year 1-interval 2-x 3-y 4-hprevious 5-hdifference 6-hrelative1 7-hrelative2 8-hrelative3 9-slope 10-aspect 11-curvatureS 12-curvatureM 13-curvatureL 14-averageRunoff1 15-averageRunoff2 16-averageRunoff3 17-discharge 18-skeletonAngleChagres 19-riverLengthChagres 20-inflowChagres 21-distChagres 22-skeletonAnglePequeni 23-riverLengthPequeni 24-inflowPequeni 25-distPequeni 26-random
col_study2 = [ 'hprevious', 'hrelative3', 'slope', 'curvatureS','curvatureM','curvatureL', 'averageRunoff1', 'averageRunoff2','averageRunoff3','discharge','skeletonAngleChagres', 'riverLengthChagres', 'distChagres', 'random']
#col_study2 = ['interval',  'hprevious', 'hrelative1','hrelative2','hrelative3', 'slope', 'aspect', 'curvature', 'dist', 'averageRunoff1', 'averageRunoff2', 'averageRunoff3', 'discharge','skeletonAngle', 'riverLength', 'inflow']
param_study = 'hdifference'

Xo2 = dfTrain[col_study2]
yo2 = dfTrain[param_study]
Xt2 = dfTest[col_study2]
yt = dfTest[param_study]
X_traino2, X_testo2, y_traino2, y_testo2 = train_test_split(Xo2, yo2, test_size=0.3, random_state=42)

"""
forest3 = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 50, bootstrap= False))

forest3.fit(X_traino2, y_traino2)
print('rf-predicting: ')
y_train_pred = forest3.predict(X_traino2)
y_test_pred = forest3.predict(X_testo2)
y_pred = forest3.predict(Xt2)
"""
def predict(alg, Xtrain, ytrain, Xpredict):
    algorithm = make_pipeline(StandardScaler(), alg)
    algorithm.fit(Xtrain, ytrain)
    prediction = algorithm.predict(Xpredict)
    return prediction

forest3 = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 50, bootstrap= False)
y_pred = predict(forest3, X_traino2, y_traino2, Xt2)

"""
print('SVR-predicting: ')
regr = SVR(C=1.0, epsilon=0.2)
y_predSVR = predict(regr, X_traino2, y_traino2, Xt2)


print('Gaussian-predicting: ')
kernel = DotProduct() + WhiteKernel()
GPR = GaussianProcessRegressor(kernel=kernel, random_state=0)
y_predGPR = predict(GPR, X_traino2, y_traino2, Xt2)


print('MLPR-predicting: ')
MLPR = MLPRegressor(alpha=1e-05, random_state=1, max_iter=500, learning_rate='adaptive', solver='sgd')
y_predMLPR = predict(MLPR, X_traino2, y_traino2, Xt2)
print('predicted')
"""

###################################
# RF IMPORTANCES
###############################
forestImportance = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 50, bootstrap= False)
features=df.columns[[4,8,9, 11, 12, 13, 14,15,16,17, 18, 19, 21, 26]]

def plotImportances(alg, feat, Xtrain, ytrain):
    alg.fit(Xtrain, ytrain)
    plt.rcParams.update({'font.size': 8})
    result = pd.DataFrame(forestImportance.feature_importances_,  df[col_study2].columns)
    result.columns = ['importance']
    result.sort_values(by='importance', ascending=False)

    print(df.head())

    importances2 = forestImportance.feature_importances_
    indices = np.argsort(importances2)
    fig1 = plt.figure()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances2[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feat[indices])
    plt.xlabel('Relative Importance')
    plt.show()
    return indices

indices = plotImportances(forestImportance, features, X_traino2, y_traino2, )


###############################
# SET WITH LESS FEATURES
##################################

col_study3 = features[indices[7:-1]]
print(col_study3)
Xo3 = dfTrain[col_study3]
yo3 = dfTrain[param_study]
Xt3 = dfTest[col_study3]
X_traino2, X_testo2, y_traino2, y_testo2 = train_test_split(Xo3, yo3, test_size=0.3, random_state=42)
forest5 = RandomForestRegressor(n_estimators= 800, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 50, bootstrap= False)
print('rf-predicting less features: ')
y_predXS = predict(forest5, X_traino2, y_traino2, Xt3)
"""
print('SVR-predicting: ')
regr = SVR(C=1.0, epsilon=0.2)
y_predSVR = predict(regr, X_traino2, y_traino2, Xt3)


print('Gaussian-predicting: ')
kernel = DotProduct() + WhiteKernel()
GPR = GaussianProcessRegressor(kernel=kernel, random_state=0)
y_predGPR = predict(GPR, X_traino2, y_traino2, Xt3)


print('MLPR-predicting: ')
MLPR = MLPRegressor(alpha=1e-05, random_state=1, max_iter=500, learning_rate='adaptive', solver='sgd')
y_predMLPR = predict(MLPR, X_traino2, y_traino2, Xt3)
print('predicted')
"""
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
    plt.show()
    return figbox

boxplot(yt, y_pred, '2012-2018 Sedimentation RF')
boxplot(yt, y_predXS, '2012-2018 Sedimentation RF less features')

"""
fig3 = plt.figure()
plt.title('2012-2018 Sedimentation RF')
plt.boxplot([y_pred, yt, (y_pred - yt)], showfliers=False)
plt.xticks([1, 2, 3], ['Prediction', 'Actual', 'Residuals'])
plt.ylabel('[m]')
plt.show()

fig3_s = plt.figure()
plt.title('2012-2018 Sedimentation RF less features')
plt.boxplot([y_predXS, yt, (y_predXS - yt)], showfliers=False)
plt.xticks([1, 2, 3], ['Prediction', 'Actual', 'Residuals'])
plt.ylabel('[m]')
plt.show()
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
    plt.show()
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
    plt.show()
    return

#plotMaps1Set(yt, y_predSVR, 'SVR Annual sedimentation 2012-2018')

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
    plt.show()
    return

plotErrorHist(yt, y_pred)

##########################
# TODO
############################
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py


####################
# Writer
####################
x_array = Xt['x'].to_numpy()
y_array = Xt['y'].to_numpy()

lines = ['Readme', 'How to write text files in Python']
outputFile =  open('C:/Users/neder/Documents/Geomatics/Unity/PCproject/DEMViewer/Assets/Output/Python_Output.txt', 'w')
for i in range(y_pred.size):
    outputFile.write(str(x_array[i]))
    outputFile.write( " ")
    outputFile.write(str(y_array[i]))
    outputFile.write(" ")
    outputFile.write(str(y_pred[i]))
    outputFile.write('\n')
outputFile.close()


