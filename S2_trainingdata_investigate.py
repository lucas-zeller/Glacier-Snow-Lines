# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:22:44 2022

@author: lzell
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime
import rasterio as rio
from rasterio.mask import mask
import pandas as pd
import geopandas as gpd
import copy
import time
import multiprocessing
import json
import geemap
from geemap import ml
import csv
import ee

from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score

#%%
# # Trigger the authentication flow.
# ee.Authenticate()

# # Initialize the library.
# ee.Initialize()

#%%
save = 0 

# define training folder path
training_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","training_data",'S2')
#training_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","training_data","old")

df_1 = pd.read_csv(os.path.join(training_folder_path,'GS2B_20200822T212529_018087_N02_training_dataset_0snow_1firn_2ice_3rock.csv'))

dfs = [df_1]
print(df_1.columns)

#%%
    
df_master = pd.concat(dfs)
print('Observations')
print('Snow:', len(df_master[df_master['type']==0]))
print('Firn:', len(df_master[df_master['type']==1]))
print('Ice: ', len(df_master[df_master['type']==2]))



#%%
# subsample per image to equal number of snow/ice/firn obs
df_master_balanced = []
c=0
for d in dfs:
    firn = d[d['type']==1]
    count = firn.shape[0]
    snow = d[d['type']==0].sample(n=count, random_state=970)
    ice = d[d['type']==2].sample(n=count, random_state=336)
    bal_df = pd.concat([firn, snow, ice])
    bal_df['image'] = c
    df_master_balanced.append(bal_df)
    c+=1
df_master_balanced = pd.concat(df_master_balanced)
df_master_balanced = df_master_balanced.apply(pd.to_numeric)

#%%
# rename bands
bands_orig = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 
              'SO_B1', 'SO_B2', 'SO_B3', 'SO_B4', 'SO_B5', 'SO_B6', 'SO_B7', 'SO_B8', 'SO_B8A', 'SO_B9', 'SO_B11', 'SO_B12']
bands_new = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2',
             'SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_re1', 'SO_re2', 'SO_re3', 'SO_nir', 'SO_re4', 'SO_vapor', 'SO_swir1', 'SO_swir2']
bands_raw = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2', 'ndwi', 'ndsi']

bands_dict = {bands_orig[i]:bands_new[i] for i in range(len(bands_orig))}
df_master_balanced = df_master_balanced.rename(columns=bands_dict)

#%%
# add ndsi, ndwi, dn
df_master_balanced['ndwi'] = (df_master_balanced['green']-df_master_balanced['nir'])/(df_master_balanced['green']+df_master_balanced['nir'])
df_master_balanced['SO_ndwi'] = (df_master_balanced['SO_green']-df_master_balanced['SO_nir'])/(df_master_balanced['SO_green']+df_master_balanced['SO_nir'])
df_master_balanced['ndsi'] = (df_master_balanced['green']-df_master_balanced['swir1'])/(df_master_balanced['green']+df_master_balanced['swir1'])
df_master_balanced['SO_ndsi'] = (df_master_balanced['SO_green']-df_master_balanced['SO_swir1'])/(df_master_balanced['SO_green']+df_master_balanced['SO_swir1'])

for i in bands_raw:
    df_master_balanced[f"DN_{i}"] = df_master_balanced[f"{i}"]/df_master_balanced[f"SO_{i}"]

# remove bad data
df_master_balanced = df_master_balanced.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
#%%

colors_to_use = ['blue','green','black']
titles = ['snow', 'firn', 'ice']

fig, axs = plt.subplots(1,3, figsize=(10,5), sharex=True, sharey=True)
for surface in [0,1,2]:
    ax = axs[surface]
    subset = df_master_balanced[df_master_balanced['type']==surface]
    ax.scatter(subset['blue'], subset['ndsi'], c=colors_to_use[surface])
    ax.set_title(titles[surface])
    ax.set_xlabel('nir')
    ax.set_ylabel('ndsi')
    ax.grid()

#%%
# test random forest. split into training/testing
predictors = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2', 'ndsi', 'ndwi',
              'DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_re1', 'DN_re2', 'DN_re3', 'DN_nir', 'DN_re4', 'DN_vapor', 'DN_swir1', 'DN_swir2', 'DN_ndsi', 'DN_ndwi']
              # 'coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2',
target = ['type']

X_train, X_test, y_train, y_test = train_test_split(df_master_balanced[predictors], df_master_balanced[target],
                                                    test_size=0.33, random_state=42)

# make our best model with the given parameters
RFC = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, 
                             min_samples_leaf=10, max_features=7, n_jobs=-1, oob_score=True, random_state = 336)
RF_fit = RFC.fit(X_train, y_train)

#%%
# option to export this to GEE
#https://geemap.org/ml/
#https://github.com/giswqs/geemap/blob/master/examples/notebooks/local_rf_training.ipynb

def rf_strings(estimator, feature_names):
    estimators = np.squeeze(estimator.estimators_)
    class_labels = estimator.classes_
    trees = []
    for e in estimators:
        s = ml.tree_to_string(e, feature_names=feature_names, labels=class_labels)
        trees.append(s)
    return trees

save = 1
if save:
    
    # convert from tree to string
    trees = rf_strings(RF_fit, predictors)
    
    # create a null geometry point for feature collection.
    null_island = ee.Geometry.Point([0, 0])

    # create a list of feature over null island
    # set the tree property as the tree string
    # encode return values (\n) as #, use to parse later
    features = [ee.Feature(null_island, {"tree": tree.replace("\n", "#")}) for tree in trees]
    
    # cast as feature collection
    fc = ee.FeatureCollection(features)
    
    # name asset
    user_id = "lzeller"
    asset_id = f"projects/{user_id}/assets/random_forest_S2"
    description = "random forest s2"
    
    # create export task and start
    task = ee.batch.Export.table.toAsset(
        collection=fc, description=description, assetId=asset_id
    )
    task.start()

    # kick off an export process so it will be saved to the ee asset
    # ml.export_trees_to_fc(trees, asset_id)
    print("Random Forest Export Started")

#%%
# check accuracy

# classify the test data with this fitted RF
y_RF_test = RF_fit.predict(X_test)
y_RF_train = RF_fit.predict(X_train)


cm_test = confusion_matrix(y_test, y_RF_test)
cm_train = confusion_matrix(y_train, y_RF_train)

print('cm train')
print(cm_train)
print('cm test')
print(cm_test)

print("OOB score:",RF_fit.oob_score_)
print("Train accuracy:",accuracy_score(y_train, y_RF_train))
print("Test accuracy:",accuracy_score(y_test, y_RF_test))

importances = pd.Series(RF_fit.feature_importances_, index=predictors)

fig, ax = plt.subplots(figsize=(8,6), dpi=100)
plt.grid(axis='y', zorder=-1)
ax.bar(range(len(predictors)),height=importances, zorder=10)
ax.set_xticks(range(len(predictors)))
ax.set_xticklabels(importances.index, rotation=60)
plt.ylabel('Importance')
plt.tight_layout()



