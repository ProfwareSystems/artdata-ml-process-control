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
import math
import datetime
import joblib
import random
import pysolar.solar as solar
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pmdarima as pm
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pio.renderers.default='browser'
pio.templates.default = "plotly_white"

# SET PREDICTION TIME:
y_name = 'dt_1hr' #Options: 'dt_1hr', 'dt_2hr', 'dt_3hr'
sarima_n_periods = 1 #Options: 1, 2, 3


# DATA PREPROCESSING

# Data source
# Read data CSV files
path = '.\data\\temp_pred'
data_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.concat((pd.read_csv(f, index_col=None, header=0) for f in data_files), axis=0, ignore_index=True)

# Data cleaning
# Drop unused data
df = df.drop(columns=['name', 'feelslike', 'preciptype', 'snow', 'snowdepth', 'visibility', 'uvindex', 'severerisk', 'conditions', 'icon', 'stations', 'precip', 'precipprob', 'windgust', 'solarradiation', 'solarenergy', 'dew', 'sealevelpressure', 'cloudcover'])
# Drop NA values
df = df.dropna()
# Convert timestamp to datetime format
df['datetime'] = pd.to_datetime(df['datetime'])
df['datetime_'] = pd.to_datetime(df['datetime'])
# Set datetime as the index
df.set_index('datetime', inplace=True)
# Sort by index
df.sort_index(inplace=True)

# Feature engineering functions

def get_solar_radiation(timestamp):
    latitude = -31.9514 # Perth coordinates (South hemisphere negative latitude)
    longitude = 115.8617 # Perth coordinates
    sun_altitude = solar.get_altitude(latitude, longitude, timestamp)
    if sun_altitude < 0 or sun_altitude > 180:
        sun_radiation = 0
    else:
        sun_radiation = math.sin(2*math.pi*sun_altitude/360)
    return sun_radiation

def get_temp_from_sarimax(data, f_periods=1):
    sarima_model = pm.auto_arima(data, 
                                 start_p=0, 
                                 start_q=0,
                                 max_p=1, 
                                 max_q=1,
                                 start_P=0,
                                 start_Q=0,
                                 max_P=1,
                                 max_Q=1,
                                 d=0, 
                                 D=1, #order of the seasonal differencing
                                 test='adf',
                                 m=24, #24 is the period of the cycle
                                 seasonal=True, #daily cycle
                                 trace=False,
                                 error_action='ignore',  
                                 suppress_warnings=True, 
                                 stepwise=True)
    sarima_pred = sarima_model.predict(n_periods=f_periods)
    return(sarima_pred[f_periods-1])

def get_temp_from_polynomial(t_2, t_1, t_0):
    n = 1
    coeff = np.polyfit([0,1,2], [t_2, t_1, t_0], n)
    return sum([pow(3,i)*coeff[n-i] for i in range(n+1)])

def random_dataset_generator(data, n):
    idx = random.randint(n, len(data)-1)
    return(data[idx-n:idx])

# Create columns for past temperatures shifted forward
df['temp_-1hr'] = df['temp'].shift(periods=1)
df['temp_-2hr'] = df['temp'].shift(periods=2)
df['temp_-3hr'] = df['temp'].shift(periods=3)

# Create dT columns for known (past) temperature change
df['dt_-3hr'] = df['temp'] - df['temp_-3hr']
df['dt_-2hr'] = df['temp'] - df['temp_-2hr']
df['dt_-1hr'] = df['temp'] - df['temp_-1hr']

# Create columns for future temperatures shifted backward
df['temp_1hr'] = df['temp'].shift(periods=-1)
df['temp_2hr'] = df['temp'].shift(periods=-2)
df['temp_3hr'] = df['temp'].shift(periods=-3)

# Create dT columns for unknown (future) temperature change
df['dt_1hr'] = df['temp_1hr'] - df['temp']
df['dt_2hr'] = df['temp_2hr'] - df['temp']
df['dt_3hr'] = df['temp_3hr'] - df['temp']

df = df.dropna()

# Temperature change statistical benchmarks - mean and std of dt
dt_abs = df['dt_1hr'].abs()
print("Mean absolute temperature change: ")
print(dt_abs.mean())
print("STD of absolute temperature change: ")
print(dt_abs.std())
print('')

# Temperature change statistical benchmarks - linear extrapolation of past 2 points
df['temp_linear_2pts_1hr'] = df.apply(lambda x: x['temp'] + x['dt_-1hr'] , axis=1)
err = df['temp_linear_2pts_1hr'] - df['temp_1hr']
sign = (df['temp_linear_2pts_1hr'] - df['temp'])*df['dt_1hr'].abs() / (df['dt_1hr']*(df['temp_linear_2pts_1hr'] - df['temp']).abs())
try:
    neg = sign.value_counts(ascending=True)[-1.0]
except:
    neg = 0 
try:
    pos = sign.value_counts(ascending=True)[1.0]
except:
    pos = 1
print('Mean error for linear extrapolation of known 2 last temperatures: ')
print(err.mean())
print("STD of error for linear extrapolation of known 2 last temperatures: ")
print(err.std())
print("Wrong direction for linear extrapolation of known 2 last temperatures: ")
print(str(int(neg * 100 / (neg + pos))) + "%")
print('')

# Temperature change statistical benchmarks - linear extrapolation of past 3 points
df['temp_linear_3pts_1hr'] = df.apply(lambda x: get_temp_from_polynomial(x['temp_-2hr'],x['temp_-1hr'],x['temp']) , axis=1)
err = df['temp_linear_3pts_1hr'] - df['temp_1hr']
sign = (df['temp_linear_3pts_1hr'] - df['temp'])*df['dt_1hr'].abs() / (df['dt_1hr']*(df['temp_linear_3pts_1hr'] - df['temp']).abs())
try:
    neg = sign.value_counts(ascending=True)[-1.0]
except:
    neg = 0 
try:
    pos = sign.value_counts(ascending=True)[1.0]
except:
    pos = 1
print('Mean error for linear extrapolation of known 3 last temperatures: ')
print(err.mean())
print("STD of error for linear extrapolation of known 3 last temperatures: ")
print(err.std())
print("Wrong direction for linear extrapolation of known 3 last temperatures: ")
print(str(int(neg * 100 / (neg + pos))) + "%")
print('')

# Temperature change statistical benchmarks - SARIMA
ref_temp = []
pred_temp = []
cur_temp = []
N=100 # Adjustable amount of samples
for i in range(N):
    #break # comment to run SARIMA performance assessment
    print("SARIMA for case N " + str(i+1) + " out of " + str(N), end="\r")
    temp = random_dataset_generator(df['temp'].tolist(), 24*7 + 1)
    ref = temp[-1]
    ref_temp.append(ref)
    cur = temp[-2]
    cur_temp.append(cur)
    pred = get_temp_from_sarimax(temp[:-1], 1)
    pred_temp.append(pred)
print("", end="\r")
sarima_df = pd.DataFrame({'ref_temp':   ref_temp,
                          'pred_temp':  pred_temp,
                          'cur_temp':   cur_temp})
err = sarima_df['pred_temp'] - sarima_df['ref_temp']
ref_dt = sarima_df['ref_temp'] - sarima_df['cur_temp']
pred_dt = sarima_df['pred_temp'] - sarima_df['cur_temp']
sign = ref_dt * pred_dt.abs() / (ref_dt.abs() * pred_dt)
try:
    neg = sign.value_counts(ascending=True)[-1.0]
except:
    neg = 0 
try:
    pos = sign.value_counts(ascending=True)[1.0]
except:
    pos = 1
print('Mean error for SARIMA: ')
print(err.mean())
print("STD of error for SARIMA: ")
print(err.std())
print("Wrong direction for SARIMA: ")
print(str(int(neg * 100 / (neg + pos))) + "%")
print('')

# Calculate solar radiation and wind vector features
# Add timezone to a new timestamp_tz column
df['datetime_tz'] = df.apply(lambda x: x['datetime_'].tz_localize(tz='Australia/Perth'), axis=1)
df['solarradiation'] = df.apply(lambda x: get_solar_radiation(x['datetime_tz']), axis=1)
df['dradiation_1hr'] = df['solarradiation'].shift(periods=-1) - df['solarradiation']

df['wind_x'] = df.apply(lambda x: x['windspeed']*math.cos(2*math.pi*x['winddir']/360), axis=1)
df['wind_y'] = df.apply(lambda x: x['windspeed']*math.sin(2*math.pi*x['winddir']/360), axis=1)

df['dwind_x_-1hr'] = df['wind_x'] - df['wind_x'].shift(periods=1)
df['dwind_y_-1hr'] = df['wind_y'] - df['wind_y'].shift(periods=1)

# Remove unused columns
df = df.drop(columns=['datetime_tz', 'datetime_', 'temp_-1hr', 'temp_-2hr', 'temp_-3hr', 'temp_1hr', 'temp_2hr', 'temp_3hr', 'temp_linear_2pts_1hr', 'temp_linear_3pts_1hr'])
# Handle missing values for calculations with row shifts
df = df.dropna(subset=['dt_1hr', 'dt_2hr', 'dt_3hr', 'dt_-1hr', 'dt_-2hr', 'dt_-3hr', 'dradiation_1hr', 'wind_x', 'wind_y'])

# Split df into day and night subsets
df_day = df[(df['dradiation_1hr'] != 0)]
df_night = df[(df['dradiation_1hr'] == 0)]
day_features = ['dradiation_1hr', 'wind_x', 'wind_y', 'humidity', 'dt_-1hr']
night_features = ['wind_x', 'wind_y', 'humidity', 'dt_-1hr']


def train_and_evaliate_models(df_slice, features, suffix, sarima_n_periods):
    # Data visaulisation
    n = int(pow(df_slice.columns.size, 0.5))
    m = int(df_slice.columns.size / n) + 1
    for plot_num in [1, 2, 3]:
        fig = make_subplots(rows=n, cols=m, subplot_titles=df_slice.columns.values)
        for i in range(n):
            for j in range(m):
                if i*m+j < df_slice.columns.size:
                    fig.add_trace(go.Scatter(x=df_slice[df_slice.columns.values[i*m+j]], y=df_slice['dt_'+str(plot_num)+'hr'], mode='markers', marker={'opacity': 0.25}),row=i+1, col=j+1)
        fig.show()
    
    # Data analysis
    # Correlation
    df_slice_labels = list(df_slice.columns.values)
    corr_mat = df_slice.corr()
    fig = px.imshow(
        corr_mat,
        x=df_slice_labels,
        y=df_slice_labels,
        zmin=-1,
        zmax=1,
        color_continuous_scale = "Balance"
    )
    fig.show()
    
    # Training dataset preparation
    # Drop unrelevant features and leave only predictors
    X = df_slice.drop(columns=[feature for feature in list(df_slice.columns.values) if feature not in features])
    y = df_slice[y_name]
    # Split dataset to training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Scale data for using by certain models
    scaler_X = StandardScaler().fit(X_train.values)
    scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
    X_train_scaled = scaler_X.transform(X_train.values)
    y_train_scaled = scaler_y.transform(y_train.values.reshape(-1, 1))
    joblib.dump(scaler_X, r".\all_models\temp_pred\scaler_X_" + suffix + '_' + y_name + ".pkl")
    joblib.dump(scaler_y, r".\all_models\temp_pred\scaler_y_" + suffix + '_' + y_name + ".pkl")
    # Show selected features
    X_labels = np.array(list(X.columns.values))
    print("Selected features list - " + suffix + ": ")
    print(X_labels)
    print('')
    
    # Models build
    
    # Gradient boosting regressor
    parameter_space = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [3, 5],
    'subsample': [0.7, 0.8, 0.9]
    }
    start_time = datetime.datetime.now()
    gb_model = GradientBoostingRegressor(random_state=0)
    gb_reg = GridSearchCV(gb_model, parameter_space, n_jobs=-1, cv=5).fit(X_train.values, y_train.values)
    print('Gradient boosting best parameters found:\n', gb_reg.best_params_)
    duration = datetime.datetime.now() - start_time
    gb_reg_score = gb_reg.score(X_test.values, y_test.values)
    joblib.dump(gb_reg, r".\all_models\temp_pred\gb_reg_model_" + suffix + '_' + y_name + ".pkl")
    print("Gradient boosting build time: ")
    print(duration)
    print("Gradient boosting regression score: ")
    print(gb_reg_score)
    pred_dt_gb = pd.Series(gb_reg.predict(X_test.to_numpy()), index=X_test.index)
    err_gb = pred_dt_gb - y_test
    sign = y_test * pred_dt_gb.abs() / (y_test.abs() * pred_dt_gb)
    try:
        neg = sign.value_counts(ascending=True)[-1.0]
    except:
        neg = 0 
    try:
        pos = sign.value_counts(ascending=True)[1.0]
    except:
        pos = 1
    print('Mean error for Gradient Boosting: ')
    print(err_gb.mean())
    print("STD of error for Gradient Boosting: ")
    print(err_gb.std())
    print("Wrong direction for Gradient Boosting: ")
    print(str(int(neg * 100 / (neg + pos))) + "%")
    print('')
    #fig = px.scatter(x=y_test, y=pred_dt_gb, title="Gradient boosting regression")
    #fig.show()
    
    # Multilayer perception regressor
    parameter_space = {
    'hidden_layer_sizes': [(200), (300), (400)], # also tested (100, 50), (100, 100), (300, 200), (300, 300)
    'activation': ['relu'], # also tested 'tanh', 'logistic'
    'solver': ['adam'], # also tested 'sgd'
    'alpha': [1e-4], # also tested  1e-3, 1e-2, 0.1, 1
    'learning_rate': ['constant', 'adaptive'],
    'early_stopping': [True, False],
    'validation_fraction': [0.1, 0.2]
    }
    start_time = datetime.datetime.now()    
    mlp_model = MLPRegressor(random_state=1, max_iter=500)
    mlp_reg = GridSearchCV(mlp_model, parameter_space, n_jobs=-1, cv=5).fit(X_train_scaled, y_train_scaled.ravel())
    print('MLP best parameters found:\n', mlp_reg.best_params_)
    duration = datetime.datetime.now() - start_time
    mlp_reg_score = mlp_reg.score(scaler_X.transform(X_test.values), scaler_y.transform(y_test.values.reshape(-1, 1)))
    joblib.dump(mlp_reg, r".\all_models\temp_pred\mlp_reg_model_" + suffix + '_' + y_name + ".pkl")
    print("MLP build time: ")
    print(duration)
    print("MLP regression score: ")
    print(mlp_reg_score)
    pred_dt_mlp_scaled = mlp_reg.predict(scaler_X.transform(X_test.values))
    pred_dt_mlp = pd.Series(scaler_y.inverse_transform(pred_dt_mlp_scaled.reshape(-1, 1))[:,0], index=X_test.index)
    err_mlp = pred_dt_mlp - y_test
    sign = y_test * pd.Series(pred_dt_mlp, index=y_test.index).abs() / (y_test.abs() * pd.Series(pred_dt_mlp, index=y_test.index))
    try:
        neg = sign.value_counts(ascending=True)[-1.0]
    except:
        neg = 0 
    try:
        pos = sign.value_counts(ascending=True)[1.0]
    except:
        pos = 1
    print('Mean error for MLP: ')
    print(err_mlp.mean())
    print("STD of error for MLP: ")
    print(err_mlp.std())
    print("Wrong direction for MLP: ")
    print(str(int(neg * 100 / (neg + pos))) + "%")
    print('')
    #fig = px.scatter(x=y_test, y=pred_dt_mlp, title="MLP regression")
    #fig.show()
    
    # SARIMA forecast
    ref_temp = []
    pred_temp = []
    cur_temp = []
    for i in range(y_test.size):
        print("SARIMA for case N " + str(i+1) + " out of " + str(y_test.size), end="\r")
        idx = df.index.get_loc(y_test.index[i])
        #idx = 0 # comment to run through y_test
        if idx-24*7>=0:
            temp = df['temp'].iloc[idx-24*7+1:idx+2].tolist() #note y_test dt_1hr is based on next temp value, hence shift by 2
            ref = temp[-1]
            ref_temp.append(ref)
            cur = temp[-2]
            cur_temp.append(cur)
            pred = get_temp_from_sarimax(temp[:-1], sarima_n_periods)
            pred_temp.append(pred)
        else:
            ref_temp.append(0)
            cur_temp.append(0)
            pred_temp.append(0)
    print("", end="\r")
    sarima_df = pd.DataFrame({'ref_temp':   ref_temp,
                              'pred_temp':  pred_temp,
                              'cur_temp':   cur_temp}, index=X_test.index)
    pred_dt_sarima = sarima_df['pred_temp'] - sarima_df['cur_temp']
    err_sarima = sarima_df['pred_temp'] - sarima_df['ref_temp']
    ref_dt = sarima_df['ref_temp'] - sarima_df['cur_temp']
    sign = ref_dt * pred_dt_sarima.abs() / (ref_dt.abs() * pred_dt_sarima)
    try:
        neg = sign.value_counts(ascending=True)[-1.0]
    except:
        neg = 0 
    try:
        pos = sign.value_counts(ascending=True)[1.0]
    except:
        pos = 1
    print('Mean error for SARIMA: ')
    print(err_sarima.mean())
    print("STD of error for SARIMA: ")
    print(err_sarima.std())
    print("Wrong direction for SARIMA: ")
    print(str(int(neg * 100 / (neg + pos))) + "%")
    print('')
    #fig = px.scatter(x=y_test, y=pred_dt_sarima, title="SARIMA prediction")
    #fig.show()
    
    # Ensemble using median selection
    pred_dt_ensemble = pd.concat([pred_dt_gb, pred_dt_mlp, pred_dt_sarima], axis=1).agg(np.median, 1)
    err_ensemble = pred_dt_ensemble - y_test
    sign = y_test * pred_dt_ensemble.abs() / (y_test.abs() * pred_dt_ensemble)
    try:
        neg = sign.value_counts(ascending=True)[-1.0]
    except:
        neg = 0 
    try:
        pos = sign.value_counts(ascending=True)[1.0]
    except:
        pos = 1
    print('Mean error for ensemble: ')
    print(err_ensemble.mean())
    print("STD of error for ensemble: ")
    print(err_ensemble.std())
    print("Wrong direction for ensemble: ")
    print(str(int(neg * 100 / (neg + pos))) + "%")
    print('')
    #fig = px.scatter(x=y_test, y=pred_dt_ensemble, title="Ensemble prediction")
    #fig.show()
    
    
train_and_evaliate_models(df_slice=df_day, features=day_features, suffix='day', sarima_n_periods=sarima_n_periods)
train_and_evaliate_models(df_slice=df_night, features=night_features, suffix='night', sarima_n_periods=sarima_n_periods)