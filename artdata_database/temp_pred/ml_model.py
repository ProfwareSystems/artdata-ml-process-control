# @author: Ivan Nemov @ Profware Systems
#
# Copyright 2023 Profware Systems - All rights reserved.
# This code is a part of ML demo case for ARTData platform.
# This is a proprietary software of Profware Systems. 
# The software copying, use, distribution, reverse engineering, disclosure and derivative works are prohibited unless explicitly allowed by Profware Systems through individual license agreement.
#

import joblib
import datetime
import pysolar.solar as solar
import math
import pmdarima as pm
import numpy as np
import datetime

def get_solar_radiation(timestamp):
    try:
        latitude = -31.9514 # Perth coordinates (South hemisphere negative latitude)
        longitude = 115.8617 # Perth coordinates
        sun_altitude = solar.get_altitude(latitude, longitude, timestamp)
        if sun_altitude < 0 or sun_altitude > 180:
            sun_radiation = 0
        else:
            sun_radiation = math.sin(2*math.pi*sun_altitude/360)
    except:
        sun_radiation = -999
    return sun_radiation

def gb_predict(predictors, quality, model_type, logging_active):
    model_names = {'day': {'1hr': 'gb_reg_model_day_dt_1hr.pkl',
                           '2hr': 'gb_reg_model_day_dt_2hr.pkl',
                           '3hr': 'gb_reg_model_day_dt_3hr.pkl'},
                 'night': {'1hr': 'gb_reg_model_night_dt_1hr.pkl',
                           '2hr': 'gb_reg_model_night_dt_2hr.pkl',
                           '3hr': 'gb_reg_model_night_dt_3hr.pkl'}}
    result = {}
    try:
        used_model_names = model_names[model_type]
        for key in list(used_model_names.keys()):
            if quality:
                model = joblib.load("/artdata/data/mpjt_000_adml_mdl/" + used_model_names[key])
                pred = model.predict(predictors)[0]
                result[key] = pred
            else:
                result[key] = -999
        if logging_active:
            ts = datetime.datetime.now()
            filename = ts.strftime('/artdata/data/mpjt_000_adml_mdl/logs/%Y-%m-%d %H-%M-%S.log')
            with open(filename, 'w') as logfile:
                logfile.write(str(predictors) + " - " + str(quality) + " - " + str(model_type) + " - " + str(result))
    except Exception as err:
        ts = datetime.datetime.now()
        filename = ts.strftime('/artdata/data/mpjt_000_adml_mdl/logs/Error-%Y-%m-%d %H-%M-%S.log')
        with open(filename, 'w') as logfile:
            logfile.write(str(predictors) + str(err))
    return result
    
def mlp_predict(predictors, quality, model_type, logging_active):
    model_names = {'day': {'1hr': 'mlp_reg_model_day_dt_1hr.pkl',
                           '2hr': 'mlp_reg_model_day_dt_2hr.pkl',
                           '3hr': 'mlp_reg_model_day_dt_3hr.pkl'},
                 'night': {'1hr': 'mlp_reg_model_night_dt_1hr.pkl',
                           '2hr': 'mlp_reg_model_night_dt_2hr.pkl',
                           '3hr': 'mlp_reg_model_night_dt_3hr.pkl'}}
    scaler_X_names = {'day': {'1hr': 'scaler_X_day_dt_1hr.pkl',
                              '2hr': 'scaler_X_day_dt_2hr.pkl',
                              '3hr': 'scaler_X_day_dt_3hr.pkl'},
                    'night': {'1hr': 'scaler_X_night_dt_1hr.pkl',
                              '2hr': 'scaler_X_night_dt_2hr.pkl',
                              '3hr': 'scaler_X_night_dt_3hr.pkl'}}
    scaler_y_names = {'day': {'1hr': 'scaler_y_day_dt_1hr.pkl',
                              '2hr': 'scaler_y_day_dt_2hr.pkl',
                              '3hr': 'scaler_y_day_dt_3hr.pkl'},
                    'night': {'1hr': 'scaler_y_night_dt_1hr.pkl',
                              '2hr': 'scaler_y_night_dt_2hr.pkl',
                              '3hr': 'scaler_y_night_dt_3hr.pkl'}}
    result = {}
    try:
        used_model_names = model_names[model_type]
        used_scaler_X_names = scaler_X_names[model_type]
        used_scaler_y_names = scaler_y_names[model_type]
        for key in list(used_model_names.keys()):
            if quality:
                model = joblib.load("/artdata/data/mpjt_000_adml_mdl/" + used_model_names[key])
                scaler_X = joblib.load("/artdata/data/mpjt_000_adml_mdl/" + used_scaler_X_names[key])
                scaler_y = joblib.load("/artdata/data/mpjt_000_adml_mdl/" + used_scaler_y_names[key])
                pred_scaled = model.predict(scaler_X.transform(predictors))[0]
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0,0]
                result[key] = pred
            else:
                result[key] = -999
        if logging_active:
            ts = datetime.datetime.now()
            filename = ts.strftime('/artdata/data/mpjt_000_adml_mdl/logs/%Y-%m-%d %H-%M-%S.log')
            with open(filename, 'w') as logfile:
                logfile.write(str(predictors) + " - " + str(quality) + " - " + str(model_type) + " - " + str(result))
    except Exception as err:
        ts = datetime.datetime.now()
        filename = ts.strftime('/artdata/data/mpjt_000_adml_mdl/logs/Error-%Y-%m-%d %H-%M-%S.log')
        with open(filename, 'w') as logfile:
            logfile.write(str(predictors) + str(err))
    return result
    
def get_temp_list(data, end_ts):
    temp = []
    data = np.flipud(data)
    bad_counter = 0
    if data[0,0] == end_ts and data[0,2] == "Good":
        last_recorded_ts = end_ts
        temp.append(data[0,1])
        for val in data:
            passed_time = ((last_recorded_ts - val[0]).seconds)//60
            if (55 <= passed_time <= 65) and val[2] == "Good":
                bad_counter = 0
                temp.append(val[1])
                last_recorded_ts = last_recorded_ts - datetime.timedelta(hours=1)
            elif passed_time > 65:
                bad_counter = bad_counter + 1
                if bad_counter > 1:
                    temp.append(-999)
                else:
                    temp.append(temp[-1])
                last_recorded_ts = last_recorded_ts - datetime.timedelta(hours=1)
    else:
        temp.append(-999)
    return list(reversed(temp))
    
def get_temp_from_sarimax(data, f_periods=[1]):
    try:
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
        sarima_pred = {}
        for f_period in f_periods:
            sarima_pred_list = sarima_model.predict(n_periods=f_period)
            sarima_pred[str(f_period)+'hr'] = sarima_pred_list[f_period-1] - data[-1]
    except:
        for f_period in f_periods:
            sarima_pred[str(f_period)+'hr'] = -999
    return(sarima_pred)
    
def get_temp_from_polynomial(ts_list, temp_list, ts, n):
    coeff = np.polyfit(ts_list, temp_list, n)
    return sum([pow(ts,i)*coeff[n-i] for i in range(n+1)])