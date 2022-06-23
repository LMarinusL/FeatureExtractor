# Morphology predictor

This project is a full functioning code as part of a thesis research on predicting sedimentation using machien learning. The code can be implemented in a Unity environment, and by running the predcition and consecutively the python scripts, predictions are made and visualised. The code can be divided into four parts:

## DEM Loader and Feature extractor
The first part in the pipeline reads a pointcloud containing a grid of popints. This is then triangulated and features correlated to morhpological processes are computed and extracted from the DEMs.

## File writer for prediction
When the features have been extracted for all DEMs, these are then written to a file where every point in the grid contains a number of features.

## Predictor
The python predictor script then reads the written file with features, uses the features of the historical data to train the ML model and predicts the sedimentation to occur. The new DEM with the predicted sediment layer is then also written to a file in the form of a grid.

## Output plotter
The prediction outpuit result along with a wide selection of machine learning quality check can be visualised with the python plotter script, however the final results of the prediction can also be checked in Unity.

# Updates and research report on theory behind the code is coming 1st of July 2022
