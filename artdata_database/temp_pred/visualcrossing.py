# @author: Ivan Nemov @ Profware Systems
#
# Copyright 2023 Profware Systems - All rights reserved.
# This code is a part of ML demo case for ARTData platform.
# This is a proprietary software of Profware Systems. 
# The software copying, use, distribution, reverse engineering, disclosure and derivative works are prohibited unless explicitly allowed by Profware Systems through individual license agreement.
#

import requests
import json
                
def get_weather_forecast(API_STR=""):
    result_data = {}
    try: 
        response = requests.request("GET", API_STR)
        jsonData = response.json()
        response.close()
        # location
        result_data['latitude'] = jsonData['latitude']
        result_data['longitude'] = jsonData['longitude']
        # current conditions
        result_data['cur_ts'] = jsonData['currentConditions']['datetimeEpoch']
        result_data['cur_temp'] = jsonData['currentConditions']['temp']
        result_data['cur_humidity'] = jsonData['currentConditions']['humidity']
        result_data['cur_precip'] = jsonData['currentConditions']['precip']
        result_data['cur_precipprob'] = jsonData['currentConditions']['precipprob']
        result_data['cur_windgust'] = jsonData['currentConditions']['windgust']
        result_data['cur_windspeed'] = jsonData['currentConditions']['windspeed']
        result_data['cur_winddir'] = jsonData['currentConditions']['winddir']
        result_data['cur_pressure'] = jsonData['currentConditions']['pressure']
        result_data['cur_cloudcover'] = jsonData['currentConditions']['cloudcover']
        result_data['cur_solarradiation'] = jsonData['currentConditions']['solarradiation']
        result_data['cur_solarenergy'] = jsonData['currentConditions']['solarenergy']
        result_data['cur_uvindex'] = jsonData['currentConditions']['uvindex']
        # forecast
        result_data['ts'] = []
        result_data['temp'] = []
        result_data['humidity'] = []
        result_data['precip'] = []
        result_data['precipprob'] = []
        result_data['windgust'] = []
        result_data['windspeed'] = []
        result_data['winddir'] = []
        result_data['pressure'] = []
        result_data['cloudcover'] = []
        result_data['solarradiation'] = []
        result_data['solarenergy'] = []
        result_data['uvindex'] = []
        for day_data in jsonData['days']:
            for hour_data in day_data['hours']:
                result_data['ts'].append(hour_data['datetimeEpoch'])
                result_data['temp'].append(hour_data['temp'])
                result_data['humidity'].append(hour_data['humidity'])
                result_data['precip'].append(hour_data['precip'])
                result_data['precipprob'].append(hour_data['precipprob'])
                result_data['windgust'].append(hour_data['windgust'])
                result_data['windspeed'].append(hour_data['windspeed'])
                result_data['winddir'].append(hour_data['winddir'])
                result_data['pressure'].append(hour_data['pressure'])
                result_data['cloudcover'].append(hour_data['cloudcover'])
                result_data['solarradiation'].append(hour_data['solarradiation'])
                result_data['solarenergy'].append(hour_data['solarenergy'])
                result_data['uvindex'].append(hour_data['uvindex'])
    except:
        jsonData = {}
    return result_data