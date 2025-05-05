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
import pandas as pd
import numpy as np
import joblib
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_objective
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pio.renderers.default='browser'
pio.templates.default = "plotly_white"


# FIXED PARAMS
feed_flow   =  100 #kg/s
ti_150      = -48 #deg_C
C_LNG       =  14 #$/MMBtu
C_CND       =  65 #$/bbl
a_1         =  0.158987 #m3/bbl
a_2         =  1055055.8526 #kJ/MMBtu
a_3         =  0.1 #-
LHV_C5      =  44938 #kJ/kg
LHV_LNG     =  48632 #kJ/kg
MW_LNG = 0.92 * 16.04 + 0.04 * 30.07 + 0.019 * 44.1 + 0.0045 * 58.12 + \
            0.0015 * 72.15 + 0.0014 * 86.18 + 0.0012 * 100.21 + 0.0045 * 58.12 + \
            0.0015 * 72.15 + 0.0064 * 14.00
MW_C5 = (0.0045 * 58.12 + 0.0015 * 72.15 + 0.0014 * 86.18 + \
         0.0012 * 100.21 + 0.0045 * 58.12 + 0.0015 * 72.15)/\
        (0.0045 + 0.0015 + 0.0014 + 0.0012 + 0.0045 + 0.0015)

# IMPORT MODELS
mlp_reg_model_Condensatedensitykgm3 = joblib.load(r".\all_models\ngl_opt\mlp_reg_model_Condensatedensitykgm3.pkl")
mlp_reg_model_Condensateflowratekgs = joblib.load(r".\all_models\ngl_opt\mlp_reg_model_Condensateflowratekgs.pkl")
mlp_reg_model_LNGC5mol = joblib.load(r".\all_models\ngl_opt\mlp_reg_model_LNGC5mol.pkl")
mlp_reg_model_LNGflowratekgs = joblib.load(r".\all_models\ngl_opt\mlp_reg_model_LNGflowratekgs.pkl")
mlp_reg_model_Longreinjectionmassflowkgs = joblib.load(r".\all_models\ngl_opt\mlp_reg_model_Longreinjectionmassflowkgs.pkl")

# IMPORT SCALERS
scaler_X = joblib.load(r".\all_models\ngl_opt\scaler_X.pkl")
scaler_y = joblib.load(r".\all_models\ngl_opt\scaler_y.pkl")

output_data_labels = ['Condensate flow rate kg/s',
                      'Condensate density kg/m3',
                      'LNG flow rate kg/s',
                      'LNG C5+ mol',
                      'Long reinjection mass flow kg/s',
                      "Reinjection mass flow kg/s"]

input_data_labels = ['Feed flow rate kg/s',
                     'Recycle ratio',
                     'TI-050 deg_C',
                     'TI-150 deg_C']

def selectiveInvTransform(scaler, data, col_name, col_names):
    dummy = pd.DataFrame(np.zeros((len(data), len(col_names))), columns=col_names)
    dummy[col_name] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy.values), columns=col_names)
    return dummy[col_name].values

def objective_eval(X):
    rec_ratio, ti_050 = X
    predictors = np.array([feed_flow, rec_ratio, ti_050, ti_150]).reshape(1, -1)
    scaled_predictors = scaler_X.transform(predictors)
    Condensateflowratekgs = selectiveInvTransform(scaler_y, 
                                                  mlp_reg_model_Condensateflowratekgs.predict(scaled_predictors),
                                                  'Condensate flow rate kg/s',
                                                  output_data_labels)[0]
    Condensatedensitykgm3 = selectiveInvTransform(scaler_y, 
                                                  mlp_reg_model_Condensatedensitykgm3.predict(scaled_predictors),
                                                  'Condensate density kg/m3',
                                                  output_data_labels)[0]
    LNGflowratekgs =        selectiveInvTransform(scaler_y, 
                                                  mlp_reg_model_LNGflowratekgs.predict(scaled_predictors),
                                                  'LNG flow rate kg/s',
                                                  output_data_labels)[0]
    LNGC5mol =              selectiveInvTransform(scaler_y, 
                                                  mlp_reg_model_LNGC5mol.predict(scaled_predictors),
                                                  'LNG C5+ mol',
                                                  output_data_labels)[0]
    Longreinjectionmassflowkgs = selectiveInvTransform(scaler_y, 
                                                  mlp_reg_model_Longreinjectionmassflowkgs.predict(scaled_predictors),
                                                  'Long reinjection mass flow kg/s',
                                                  output_data_labels)[0]
    # value of C5+ in LNG
    G_LNG_C5 = LNGflowratekgs * LNGC5mol * (MW_C5 / MW_LNG) * LHV_C5 * C_LNG / a_2
    # value of C5+ in Condensate
    G_CND = (Condensateflowratekgs / (Condensatedensitykgm3 * a_1)) * C_CND
    # value of LNG losses due to warm LPG reinjection
    G_LNG_LOSS = - Longreinjectionmassflowkgs * a_3 * LHV_LNG * C_LNG / a_2
    return([G_LNG_C5, G_CND, G_LNG_LOSS])

def objective(X):
    return -sum(objective_eval(X))

space = [Real(0.025, 0.175, name='rec_ratio'),
         Real(-20.0, 0, name='ti_050')]

result = gp_minimize(func=objective,
                     dimensions=space,
                     acq_func="EI",      # Expected Improvement
                     n_calls=30,
                     random_state=42
                     )

# Extract results
rec_ratio_opt, ti_050_opt = result.x
G = -result.fun

print(f"Maximum value of profit function ≈ {G:.4f} at (rec_ratio, ti_050) ≈ ({rec_ratio_opt:.4f}, {ti_050_opt:.4f})")
_ = plot_objective(result)


df_obj_list = []
for i in range(16):
    a = objective_eval([0.025+i/100, -17])
    df_obj_list.append([0.025+i/100, a[0], a[1], a[2]])

df_obj = pd.DataFrame(df_obj_list, columns=['Reflux ratio', 'G_LNG_C5 $/sec', 'G_CND $/sec', 'G_LNG_LOSS $/sec'])
df_obj['G $/sec'] = df_obj['G_LNG_C5 $/sec'] + df_obj['G_CND $/sec'] + df_obj['G_LNG_LOSS $/sec']

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df_obj['Reflux ratio'], y=df_obj['G_LNG_C5 $/sec'], name='G_LNG_C5 $/sec'), secondary_y=False)
fig.add_trace(go.Scatter(x=df_obj['Reflux ratio'], y=df_obj['G_CND $/sec'], name='G_CND $/sec'), secondary_y=False)
fig.add_trace(go.Scatter(x=df_obj['Reflux ratio'], y=df_obj['G_LNG_LOSS $/sec'], name='G_LNG_LOSS $/sec'), secondary_y=False)
fig.add_trace(go.Scatter(x=df_obj['Reflux ratio'], y=df_obj['G $/sec'], name='G $/sec'), secondary_y=True)
fig.update_yaxes(title_text='$/sec')
fig.update_xaxes(title_text='Reflux ratio')  
fig.show()