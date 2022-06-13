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


##############
import pathlib
print(pathlib.Path(__file__).parent.resolve())
df = pd.read_csv('C:/Users/neder/Documents/Geomatics/Unity/PCproject/DEMViewer/Assets/Output/outputGridFull.txt', sep=" ")

####################
# Writer
####################

svrModel = SVR(C=1.0, epsilon=0.2)
col_study_peq = [ 'depth', 'hrelative3','hrelative2', 'slope', 'curvatureL', 'averageRunoff1', 'averageRunoff2','averageSlope','riverLengthPequeni', 'distPequeni','totalDistPequeni']
col_study_chag = [ 'depth', 'riverLengthChagres','totalDistChagres', 'distChagres','averageRunoff3', 'averageRunoff2', 'averageRunoff1','aspect', 'averageSlope','skeletonAngleChagres','hrelative3','hrelative2','slope', 'curvatureL']

def predict(alg, Xtrain, ytrain, Xpredict):
    algorithm = make_pipeline(StandardScaler(), alg)
    algorithm.fit(Xtrain, ytrain)
    prediction = algorithm.predict(Xpredict)
    return prediction

dfpred = pd.read_csv('C:/Users/neder/Documents/Geomatics/Unity/PCproject/DEMViewer/Assets/Output/outputGridPredParams.txt', sep=" ")
X_arr = dfpred[col_study_chag][dfpred.y < -(9.5 / 2) * dfpred.x + 4545 ][dfpred.y > dfpred.x + 50][dfpred.y < 920][ dfpred.y > -1.25 * dfpred.x + 1575 ][ dfpred.y > 630 ][ dfpred.y < 970].to_numpy()
X_arr_sample = dfpred[col_study_chag][dfpred.y < -(9.5 / 2) * dfpred.x + 4545 ][dfpred.y > dfpred.x + 50][dfpred.y < 920][ dfpred.y > -1.25 * dfpred.x + 1575 ][ dfpred.y > 630 ][ dfpred.y < 970]
print("pred file read")
dfTrainFull = sklearn.utils.resample(df[df.year < 2013][df.hdifference > -2][df.hdifference < 10][df.y < -(9.5 / 2) * df.x + 4545 ][df.y > df.x + 50][df.y < 920][ df.y > -1.25 * df.x + 1575 ][ df.y > 630 ][ df.y < 970], n_samples=10000, random_state=None, stratify=None)
set_X_T = dfTrainFull[col_study_chag]
set_y_T = dfTrainFull['hdifference']

pred = predict(svrModel, set_X_T, set_y_T, X_arr)
index_array = dfpred['index'][dfpred.y < -(9.5 / 2) * dfpred.x + 4545 ][dfpred.y > dfpred.x + 50][dfpred.y < 920][ dfpred.y > -1.25 * dfpred.x + 1575 ][ dfpred.y > 630 ][ dfpred.y < 970].to_numpy()
x_array =         dfpred['x'][dfpred.y < -(9.5 / 2) * dfpred.x + 4545 ][dfpred.y > dfpred.x + 50][dfpred.y < 920][ dfpred.y > -1.25 * dfpred.x + 1575 ][ dfpred.y > 630 ][ dfpred.y < 970].to_numpy()
y_array =         dfpred['y'][dfpred.y < -(9.5 / 2) * dfpred.x + 4545 ][dfpred.y > dfpred.x + 50][dfpred.y < 920][ dfpred.y > -1.25 * dfpred.x + 1575 ][ dfpred.y > 630 ][ dfpred.y < 970].to_numpy()
d_array =    dfpred['height'][dfpred.y < -(9.5 / 2) * dfpred.x + 4545 ][dfpred.y > dfpred.x + 50][dfpred.y < 920][ dfpred.y > -1.25 * dfpred.x + 1575 ][ dfpred.y > 630 ][ dfpred.y < 970].to_numpy()
path_array = dfpred['distChagres'][dfpred.y < -(9.5 / 2) * dfpred.x + 4545 ][dfpred.y > dfpred.x + 50][dfpred.y < 920][ dfpred.y > -1.25 * dfpred.x + 1575 ][ dfpred.y > 630 ][ dfpred.y < 970].to_numpy()
print("pred made")


outputFile =  open('C:/Users/neder/Documents/Geomatics/Unity/PCproject/DEMViewer/Assets/Output/Python_Output.txt', 'w')
factor = 0.4
for i in range(pred.size):
    outputFile.write(str(index_array[i]))
    outputFile.write( " ")
    outputFile.write(str(x_array[i]))
    outputFile.write( " ")
    outputFile.write(str(y_array[i]))
    outputFile.write(" ")
    outputFile.write(str(((pred[i] * factor)+ d_array[i] )))
    outputFile.write('\n')
outputFile.close()
print("output written")

def plotPred(x_vals, y_vals, pred_vals, title, min, max, range):
    plt.rcParams.update({'font.size': 20})
    fig5, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                                        figsize=(20, 10))
    cm = plt.cm.get_cmap('RdYlGn', range)
    sc = ax.scatter(x_vals, y_vals,
            linewidths=1, alpha=.7,
                edgecolor='none',
            s = 20,
            c=(pred_vals),
                cmap=cm,  vmin=min, vmax=max)
    cbar = fig5.colorbar(sc)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.tick_params(labelsize=12)
    fig5.subplots_adjust(wspace=0.03, hspace=0.05)
    fig5.suptitle(title)
    plt.draw()


#plotPred(x_array, y_array, d_array + pred, 'height', 50, 80, 7)
plotPred(x_array, y_array, pred, 'difference', -2, 2, 5)
plotPred(x_array, y_array, path_array, 'path', 0, 30, 5)


plt.show()