# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:47:16 2025
"""
# @author: Ivan Nemov @ Profware Systems
#
# Copyright 2023 Profware Systems - All rights reserved.
# This code is a part of ML demo case for ARTData platform.
# This is a proprietary software of Profware Systems. 
# The software copying, use, distribution, reverse engineering, disclosure and derivative works are prohibited unless explicitly allowed by Profware Systems through individual license agreement.
#
import os
import glob
import pandas as pd
import numpy as np
import datetime
import joblib
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pio.renderers.default='browser'
pio.templates.default = "plotly_white"


# DATA PREPROCESSING

# Data source
# Read data CSV files
path = '.\data\\ngl_opt'
data_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.concat((pd.read_csv(f, index_col=None, header=5) for f in data_files), axis=0, ignore_index=True)

# Data cleaning
# Drop NA values
df_raw = df.dropna()
# Filter out bad status data
df = df_raw[(df_raw['Status'] == True)]
df = df.drop(columns=['Status'])
#df_temp = df[(df['Condensate density kg/m3'] < 684) & (df['LNG C5+ mol'] < 0.0015)]
# Fileter out ouliers
try:
    outlier_idx = joblib.load(r".\ngl_outlier_idx.pkl")
    df = df.drop(outlier_idx)
except:
    pass

# Calculate recycle to feed ratio
df['Recycle ratio'] = df['Recycle flow rate kg/s'] / df['Feed flow rate kg/s']
df = df.drop(columns=['Recycle flow rate kg/s'])
df = df.iloc[:, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8]]

output_data_labels = ['Condensate flow rate kg/s',
                      'Condensate density kg/m3',
                      'LNG flow rate kg/s',
                      'LNG C5+ mol',
                      'Long reinjection mass flow kg/s',
                      'Reinjection mass flow kg/s']

input_data_labels = ['Feed flow rate kg/s',
                     'Recycle ratio',
                     'TI-050 deg_C',
                     'TI-150 deg_C']

# Data visaulisation
n = int(pow(df.columns.size, 0.5))
m = math.ceil(int(df.columns.size / n))
subplot_titles = input_data_labels + output_data_labels
for plot_num, plot_name in enumerate(output_data_labels):
    subplots = subplot_titles.copy()
    subplots.remove(output_data_labels[plot_num])
    fig = make_subplots(rows=n, cols=m)
    for i in range(n):
        for j in range(m):
            if i*m+j < len(subplots):
                fig.add_trace(go.Scatter(x=df[subplots[i*m+j]], y=df[output_data_labels[plot_num]], mode='markers', marker={'opacity': 0.25}), row=i+1, col=j+1)
                fig.update_xaxes(title_text=subplots[i*m+j], row=i+1, col=j+1)  
    fig.update_layout(title_text=output_data_labels[plot_num])
    fig.show()

# Data analysis
# Correlation
df_labels = list(df.columns.values)
corr_mat = df.corr()
fig = px.imshow(
    corr_mat,
    x=df_labels,
    y=df_labels,
    zmin=-1,
    zmax=1,
    color_continuous_scale = "Balance"
)
fig.show()

# ML MODEL TRAINING

# Training dataset preparation
# Drop unrelevant features and leave only predictors
X = df[input_data_labels]
y = df[output_data_labels]
# Split dataset to training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Scale data for using by certain models
scaler_X = StandardScaler().fit(X_train.values)
scaler_y = StandardScaler().fit(y_train.values)
X_train_scaled = scaler_X.transform(X_train.values)
y_train_scaled = scaler_y.transform(y_train.values)
joblib.dump(scaler_X, r".\all_models\ngl_opt\scaler_X.pkl")
joblib.dump(scaler_y, r".\all_models\ngl_opt\scaler_y.pkl")


def selectiveTransform(scaler, data, col_name, col_names):
    dummy = pd.DataFrame(np.zeros((len(data), len(col_names))), columns=col_names)
    dummy[col_name] = data
    dummy = pd.DataFrame(scaler.transform(dummy.values), columns=col_names)
    return dummy[col_name].values.reshape(-1, 1)


def selectiveInvTransform(scaler, data, col_name, col_names):
    dummy = pd.DataFrame(np.zeros((len(data), len(col_names))), columns=col_names)
    dummy[col_name] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy.values), columns=col_names)
    return dummy[col_name].values


# Multilayer perception regressor
n = int(pow(len(output_data_labels), 0.5))
m = math.ceil(int(len(output_data_labels) / n))
subplot_titles = output_data_labels
fig = make_subplots(rows=n, cols=m, subplot_titles=subplot_titles)
fig.update_layout(title_text="MLP regression - prediction vs actual values")
for idx, target_name in enumerate(output_data_labels):
    print('Training MLP model for "' + target_name + '"...')
    parameter_space = {
    "Condensate flow rate kg/s":
    {
    'hidden_layer_sizes': [(500)], 
    'activation': ['tanh'],
    'solver': ['adam'],
    'alpha': [0.1],
    'learning_rate': ['constant'],
    'early_stopping': [True],
    'validation_fraction': [0.1]
    },
    "Condensate density kg/m3":
    {
    'hidden_layer_sizes': [(500)], 
    'activation': ['tanh'], 
    'solver': ['adam'], 
    'alpha': [0.1], 
    'learning_rate': ['constant'],
    'early_stopping': [True],
    'validation_fraction': [0.1]
    },
    "LNG flow rate kg/s":
    {
    'hidden_layer_sizes': [(500)], 
    'activation': ['tanh'], 
    'solver': ['adam'], 
    'alpha': [0.1], 
    'learning_rate': ['constant'],
    'early_stopping': [True],
    'validation_fraction': [0.1]
    },
    "LNG C5+ mol":
    {
    'hidden_layer_sizes': [(400)], 
    'activation': ['tanh'], 
    'solver': ['adam'], 
    'alpha': [0.1], 
    'learning_rate': ['constant'],
    'early_stopping': [True],
    'validation_fraction': [0.1]
    },
    "Long reinjection mass flow kg/s":
    {
    'hidden_layer_sizes': [(500)], 
    'activation': ['tanh'], 
    'solver': ['adam'], 
    'alpha': [0.1], 
    'learning_rate': ['constant'],
    'early_stopping': [True],
    'validation_fraction': [0.1]
    },
    "Reinjection mass flow kg/s":
    {
    'hidden_layer_sizes': [(500)], 
    'activation': ['tanh'], 
    'solver': ['adam'], 
    'alpha': [0.1], 
    'learning_rate': ['constant'],
    'early_stopping': [True],
    'validation_fraction': [0.1]
    },
    }
    start_time = datetime.datetime.now()    
    mlp_model = MLPRegressor(random_state=1, max_iter=500)
    mlp_reg = GridSearchCV(mlp_model, parameter_space[target_name], n_jobs=-1, cv=5).fit(X_train_scaled, y_train_scaled[:,idx].ravel())
    print('MLP best parameters found:\n', mlp_reg.best_params_)
    duration = datetime.datetime.now() - start_time
    mlp_reg_score = mlp_reg.score(scaler_X.transform(X_test.values),
                                  selectiveTransform(scaler_y, y_test[target_name].values,
                                  target_name,
                                  output_data_labels))
    filename = r".\all_models\ngl_opt\mlp_reg_model_" + "".join(x for x in target_name if x.isalnum()) + ".pkl"
    joblib.dump(mlp_reg, filename)
    print("MLP build time: ")
    print(duration)
    print("MLP regression score: ")
    print(mlp_reg_score)
    pred_dt_mlp_scaled = mlp_reg.predict(scaler_X.transform(X_test.values))
    pred_dt_mlp = pd.Series(selectiveInvTransform(scaler_y, pred_dt_mlp_scaled, target_name, output_data_labels), index=X_test.index)
    err_mlp = pred_dt_mlp - y_test[target_name]
    print('Mean error for MLP: ')
    print(err_mlp.mean())
    print("STD of error for MLP: ")
    print(err_mlp.std())
    print('')
    fig.add_trace(go.Scatter(x=y_test[target_name], y=pred_dt_mlp, mode='markers', marker={'opacity': 0.25}), row=int(idx/m)+1, col=idx-int(idx/m)*m+1)

fig.show()