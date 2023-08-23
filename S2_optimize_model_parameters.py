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
import pandas as pd
import copy
import time
import geemap
from geemap import ml
import ee

from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, f1_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

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

df_1 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20190901T200911_021904_N02_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_2 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20200909T194951_027252_N02_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_3 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20210829T211521_032315_N03_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_4 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20220810T213541_037263_N04_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_5 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20220918T202121_037820_N04_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_6 = pd.read_csv(os.path.join(training_folder_path,'GS2B_20190822T210029_012853_N02_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_7 = pd.read_csv(os.path.join(training_folder_path,'GS2B_20200822T212529_018087_N02_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_8 = pd.read_csv(os.path.join(training_folder_path,'GS2B_20210815T204019_023206_N03_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))

dfs = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8]
# print(df_1.columns)

#%%

for i in range(len(dfs)):
    dfs[i]['image'] = i
df_all = pd.concat(dfs)
# print('Observations')
# print('Snow:', len(df_master[df_master['type']==1]))
# print('Firn:', len(df_master[df_master['type']==2]))
# print('Ice: ', len(df_master[df_master['type']==3]))


# # subsample per image to equal number of snow/ice/firn obs
# balance = 0

# df_master_balanced = []
# c=0
# for d in dfs:
    
#     if balance:
#         firn = d[d['type']==2]
#         count = firn.shape[0]
#         snow = d[d['type']==1].sample(n=count, random_state=970)
#         ice = d[d['type']==3].sample(n=count, random_state=970)
#         rock = d[d['type']==4].sample(n=count, random_state=970)
#         shadow = d[d['type']==5].sample(n=count, random_state=970)
#         water = d[d['type']==6].sample(n=count, random_state=970)
#         bal_df = pd.concat([firn, snow, ice, rock, shadow, water])
#     else:
#         bal_df = d
    
#     bal_df['image'] = c
#     df_master_balanced.append(bal_df)
#     c+=1
    
# df_master_balanced = pd.concat(df_master_balanced)
# df_master_balanced = df_master_balanced.apply(pd.to_numeric)


# rename bands
bands_orig = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 
              'SO_B1', 'SO_B2', 'SO_B3', 'SO_B4', 'SO_B5', 'SO_B6', 'SO_B7', 'SO_B8', 'SO_B8A', 'SO_B9', 'SO_B11', 'SO_B12']
bands_new = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2',
             'SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_re1', 'SO_re2', 'SO_re3', 'SO_nir', 'SO_re4', 'SO_vapor', 'SO_swir1', 'SO_swir2']
bands_raw = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2', 'ndwi', 'ndsi']

bands_dict = {bands_orig[i]:bands_new[i] for i in range(len(bands_orig))}
df_all = df_all.rename(columns=bands_dict)

# add ndsi, ndwi, dn
df_all['ndwi'] = (df_all['green']-df_all['nir'])/(df_all['green']+df_all['nir'])
df_all['SO_ndwi'] = (df_all['SO_green']-df_all['SO_nir'])/(df_all['SO_green']+df_all['SO_nir'])
df_all['ndsi'] = (df_all['green']-df_all['swir1'])/(df_all['green']+df_all['swir1'])
df_all['SO_ndsi'] = (df_all['SO_green']-df_all['SO_swir1'])/(df_all['SO_green']+df_all['SO_swir1'])
# df_all['nir2swir'] = (df_all['nir']*df_all['nir'])/(df_all['swir1'])
# df_all['SO_nir2swir'] = (df_all['SO_nir']*df_all['SO_nir'])/(df_all['SO_swir1'])


for i in bands_raw:
    df_all[f"DN_{i}"] = df_all[f"{i}"]/df_all[f"SO_{i}"]

# remove bad data
df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

#%%
## quick figure to investigate the spectral differences of each class
titles = ['snow', 'firn', 'ice', 'rock', 'shadow', 'water']
x_var = 'ndwi'
y_var = 'DN_re1'

x_range = np.nanpercentile(df_all[x_var], [1,99])
y_range = np.nanpercentile(df_all[y_var], [1,99])
all_range = [x_range,y_range]

fig, axs = plt.subplots(1,6, figsize=(14,4), sharex=True, sharey=True)
for surface in [0,1,2,3,4,5]:
    ax = axs[surface]
    subset = df_all[df_all['type']==surface+1]
    
    # ax.scatter(subset[x_var], subset[y_var], c=colors_to_use[surface])
    ax.hist2d(subset[x_var], subset[y_var], bins=100, density=False, range=all_range, cmap='RdPu')
    
    ax.set_title(titles[surface])
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.grid()
plt.tight_layout()

#%%
# test random forest. split into training/testing
predictors = [
                 'coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2', 'ndsi', 'ndwi',
                'SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_re1', 'SO_re2', 'SO_re3', 'SO_nir', 'SO_re4', 'SO_vapor', 'SO_swir1', 'SO_swir2', 'SO_ndsi', 'SO_ndwi',
                'DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_re1', 'DN_re2', 'DN_re3', 'DN_nir', 'DN_re4', 'DN_vapor', 'DN_swir1', 'DN_swir2', 'DN_ndsi', 'DN_ndwi',
                # 'nir2swir',
              # 're1', 'coastal', 'ndsi', 'ndwi', 'DN_ndsi', 'SO_vapor', 'SO_ndsi'
              ]
              
target = 'type'

### lets iterate through having all 8 images being the 'test' and save the results for each

# list to save results
results_df = []
importances_df = []
all_cms_test = []
all_cms_train = []

# we can optimize our classifier by testing out different parameters for it
n_es = [5,10,15,30,50] # n_estimators (number of trees)
m_ds = [5,10,15,30,50] # max_depth (depth of each tree)

for n_e in n_es:
    for m_d in m_ds:
        for i in df_all['image'].unique():
            
            test_image = i
            print(f"Working on: {n_e},{m_d} - test image {i}")
            
            # subset your training data
            df_train = df_all[df_all['image']!=test_image]
            X_train = df_train[predictors]
            y_train = df_train[target]
            
            # subset your testing data
            df_test = df_all[df_all['image']==test_image]
            X_test = df_test[predictors]
            y_test = df_test[target]
            
            # set class weights for the classifier
            weights = {1:1, 2:1, 3:1, 4:0.5, 5:0.2, 6:0.2}
            # weights = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1}
            
            # initiate and fit the classifier
            RFC = RandomForestClassifier(n_estimators=n_e, max_depth=m_d, min_samples_split=500, min_samples_leaf=100,
                                         class_weight=weights, max_features='sqrt', n_jobs=-1, oob_score=True, random_state = 336)
            RF_fit = RFC.fit(X_train, y_train)
        
            ### check accuracy
            # classify the test data with this fitted RF
            y_RF_test = RF_fit.predict(X_test)
            y_RF_train = RF_fit.predict(X_train)
            
            # create confusion matrices
            cm_test = confusion_matrix(y_test, y_RF_test)
            cm_train = confusion_matrix(y_train, y_RF_train)
            
            # get oob score, accuracy, etc...
            oob = RF_fit.oob_score_
            acc_train = accuracy_score(y_train, y_RF_train)
            acc_test = accuracy_score(y_test, y_RF_test)
            
            # predictor importances
            importances = pd.Series(RF_fit.feature_importances_, index=predictors)
            
            
            print_results = 0
            if print_results:
                print('cm train')
                print(['Snow', 'Firn', 'Ice', 'Deb.', 'Shad.', 'W'])
                print(cm_train)
                
                print()
                print('cm test')
                print(['Snow', 'Firn', 'Ice', 'D', 'S', 'W'])
                print(cm_test)
                
                print()
                print("OOB score:",oob)
                print("Train accuracy:",acc_train)
                print("Test accuracy:",acc_test)
                # print("Test F1 score:",f1_score(y_test, y_RF_test, average='weighted') )
            
            
            make_importances_figures = 0
            if make_importances_figures:
                fig, ax = plt.subplots(figsize=(9,5), dpi=100)
                ax.bar(range(len(predictors)),height=importances, zorder=10)
                
                plt.grid(axis='y', zorder=1)
                ax.set_xticks(range(len(predictors)))
                ax.set_xticklabels(importances.index, rotation=60)
                
                plt.ylabel('Importance')
                plt.title(f'Test Image: {test_image}')
                plt.tight_layout()
                
            ### save everything
            results_i = [n_e, m_d, i, oob, acc_train, acc_test]
            results_df.append(results_i)
            importances_df.append(importances)
            all_cms_test.append(cm_test)
            all_cms_train.append(cm_train)

results_df = pd.DataFrame(results_df, columns=['n_estimators', 'max_depths', 'image', 'oob', 'accuracy_train', 'accuracy_test'])

#%%
### lets go back through the test_cms and recalculate a new accuracy metric by
### class1=snow, class2=firn,ice,water,debris
snow_accs = []
for cm in all_cms_train:
    cm2 = cm.copy()
    cm = np.delete(cm, (4), axis=0) # delete shadow rows/columns
    cm = np.delete(cm, (4), axis=1)
    cm_new = np.array([ [ np.sum(cm[0,0])  ,  np.sum(cm[0,1:])  ],
                        [ np.sum(cm[1:,0]) ,  np.sum(cm[1:,1:]) ] ])
    
    snow_acc = (cm_new[0,0] + cm_new[1,1])/np.nansum(cm_new)
    snow_accs.append(snow_acc)

results_df['accuracy_test_snow'] = snow_accs
#%%
save_csv=1
if save_csv:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(training_folder_path, 'training_results', f"{now}.csv" )
    results_df.to_csv(out_path, index=False)

#%%
### lets make some figures to investigate our findings
# max_depth:15, n_estimators:50 looks good
measures = ['accuracy_train', 'accuracy_test', 'oob', 'accuracy_test_snow']
parameters = [n_es, m_ds]
p_names = ['n_estimators', 'max_depths']

# initiate figure with correct size and number of axes
fig,axs = plt.subplots(len(parameters),len(measures), figsize=(len(measures)*3, len(parameters)*3))

# each accuracy measure gets its own column
c=0 # keep track of columns
for m in measures:
    r=0 # keep track of rows

    # each parameter (n_estimators, max_depth) gets its own row
    for p in parameters:
        p_name = p_names[r] # get the parameter name
        ax = axs[r,c] # grab the correct axis
        
        # for each value of this parameter, get the average of this accuracy metric and plot them
        to_plot = []
        for v in p:
            
            # subset to these values, average, append
            subset_df = results_df[results_df[p_name]==v]
            avg = np.nanmedian(subset_df[m])
            to_plot.append(avg)
            
            # also scatter each individual obs in the background
            ax.scatter( [v]*len(subset_df), subset_df[m], s=1, c='grey')
            
        # scatter the averages in this subplots
        ax.scatter(p, to_plot, c='black', label=m)
        
        # titles and labels
        # ax.set_title(m)
        ax.set_xlabel(p_name)
        ax.set_ylabel(m)
        
        # iterate row/column
        r+=1
    c+=1

plt.tight_layout()

#%%
### train RF just to id shadows
# test random forest. split into training/testing
# predictors = [
#               #   'coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2', 'ndsi', 'ndwi',
#               # 'SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_re1', 'SO_re2', 'SO_re3', 'SO_nir', 'SO_re4', 'SO_vapor', 'SO_swir1', 'SO_swir2', 'SO_ndsi', 'SO_ndwi',
#               # 'DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_re1', 'DN_re2', 'DN_re3', 'DN_nir', 'DN_re4', 'DN_vapor', 'DN_swir1', 'DN_swir2', 'DN_ndsi', 'DN_ndwi',
#               're1', 'coastal', 'ndsi', 'ndwi', 'DN_ndsi', 'SO_vapor', 'SO_ndsi'
#               ]

# target = ['type']

# df_shadow = df_master_balanced.copy()
# types = df_shadow['type']
# types[types!=5] = 0
# types[types==5] = 1
# df_shadow['type'] = types

# X_train, X_test, y_train, y_test = train_test_split(df_shadow[predictors], df_shadow[target],
#                                                     test_size=0.33, random_state=42)

# # make our best model with the given parameters
# weights = {1:1, 2:1, 3:1, 4:0.5, 5:1, 6:0.5}
# RFC = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=100, min_samples_leaf=10,
#                              max_features=15, n_jobs=-1, oob_score=True, random_state = 336)
# RF_fit = RFC.fit(X_train, y_train)




# # check accuracy

# # classify the test data with this fitted RF
# y_RF_test = RF_fit.predict(X_test)
# y_RF_train = RF_fit.predict(X_train)


# cm_test = confusion_matrix(y_test, y_RF_test)
# cm_train = confusion_matrix(y_train, y_RF_train)

# print('cm train')
# print(cm_train)
# print('cm test')
# print(cm_test)

# print("OOB score:",RF_fit.oob_score_)
# print("Train accuracy:",accuracy_score(y_train, y_RF_train))
# print("Test accuracy:",accuracy_score(y_test, y_RF_test))

# importances = pd.Series(RF_fit.feature_importances_, index=predictors)

# fig, ax = plt.subplots(figsize=(8,6), dpi=100)
# plt.grid(axis='y', zorder=-1)
# ax.bar(range(len(predictors)),height=importances, zorder=10)
# ax.set_xticks(range(len(predictors)))
# ax.set_xticklabels(importances.index, rotation=60)
# plt.ylabel('Importance')
# plt.tight_layout()


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

save = 0
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
    asset_id = f"projects/{user_id}/assets/random_forest_S2_TOA"
    description = "random forest s2"
    
    # create export task and start
    task = ee.batch.Export.table.toAsset(
        collection=fc, description=description, assetId=asset_id
    )
    task.start()

    # kick off an export process so it will be saved to the ee asset
    # ml.export_trees_to_fc(trees, asset_id)
    print("Random Forest Export Started")

