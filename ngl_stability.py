# -*- coding: utf-8 -*-
"""
Created on Mon May 19 18:58:59 2025
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
import os
import glob
import itertools
from scipy.interpolate import griddata
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pio.renderers.default='browser'
pio.templates.default = "plotly_white"
import sys


def selectiveTransform(scaler, data, col_name, col_names):
    dummy = pd.DataFrame(np.zeros((len(data), len(col_names))), columns=col_names)
    dummy[col_name] = data
    dummy = pd.DataFrame(scaler.transform(dummy.values), columns=col_names)
    return dummy[col_name].values.reshape(-1, 1)

#           PREPARE DATA AND MODEL

# IMPORT RAW DATA
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
# Calculate recycle to feed ratio
df['Recycle ratio'] = df['Recycle flow rate kg/s'] / df['Feed flow rate kg/s']
df = df.drop(columns=['Recycle flow rate kg/s'])
df = df.iloc[:, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8]]

# IMPORT MODEL
mlp_reg_model_LNGC5mol = joblib.load(r".\all_models\ngl_opt\mlp_reg_model_LNGC5mol.pkl")

# IMPORT SCALERS
scaler_X = joblib.load(r".\all_models\ngl_opt\scaler_X.pkl")
scaler_Y = joblib.load(r".\all_models\ngl_opt\scaler_y.pkl")

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

#           CALCULATE LIPSCHITZ CONSTANT UPPER BOUND

# Extract weights
W1 = mlp_reg_model_LNGC5mol.best_estimator_.coefs_[0]  # input to hidden
W2 = mlp_reg_model_LNGC5mol.best_estimator_.coefs_[1]  # hidden to output

# Euclidean norm (2-norm) of weights in MLP layers
W1_norm = np.linalg.norm(W1, ord=2)
W2_norm = np.linalg.norm(W2, ord=2)

# Partial Lipschitz upper bounds per input
L_partial_bound = []

for i in range(W1.shape[0]):  # For each input feature
    input_norm = np.linalg.norm(W1[i, :], ord=2)  # Row i of W1 corrsponds to input i
    L_i = input_norm * W2_norm
    L_partial_bound.append(L_i)

# Overall Lipschitz upper bound
L_model = W1_norm * W2_norm

print("Analytical partial Lipschitz upper bounds per input:", np.round(L_partial_bound, 3))
print("Analytical Lipschitz upper bound:", np.round(L_model, 3))


#           CALCULATE DATASET INPUT LIMITS
input_limits = []
for input_data_label in input_data_labels:
    input_limits.append([df[input_data_label].min(), df[input_data_label].max()])
scaled_input_limits = scaler_X.transform(np.transpose(np.array(input_limits)))
print("Input scaled limits:", scaled_input_limits)


#           CALCULATE LIPSCHITZ CONSTANT LOWER BOUND

n_points = 5000 # number of points
n_neighbours = 200 # number of steps from a point
X = scaler_X.transform(df[input_data_labels].values)
Y = selectiveTransform(scaler_Y, df['LNG C5+ mol'].values, 'LNG C5+ mol', output_data_labels)
idx_points = np.random.randint(low=0, high=X.shape[0], size=n_points)
X = X[idx_points]
Y = Y[idx_points].ravel()
X_max = 0 # inputs where maximum dY/dX is observed
dY_dX_max = 0 # maximum dY/dX value
for i in range(n_points):
    idx_neighbours = np.random.randint(low=0, high=X.shape[0], size=n_neighbours)
    idx_neighbours = idx_neighbours[idx_neighbours!=i]
    dX = X[idx_neighbours] - X[i]
    dY = Y[idx_neighbours] - Y[i]
    dX_norm2 = np.linalg.norm(dX, ord=2, axis=1)
    valid_idx = np.where(dX_norm2!=0)
    dY_dX = np.max(np.abs(dY[valid_idx]/dX_norm2[valid_idx]))
    if dY_dX > dY_dX_max:
        dY_dX_max = dY_dX
        X_max = X[i]

print("Lipschitz lower bound from training data is", np.round(dY_dX_max, 3), "at X", np.round(X_max, 3))


#           CALCULATE EMPIRICAL LIPSCHITZ CONSTANT

X_limits = np.array([-np.abs(scaled_input_limits).max(), np.abs(scaled_input_limits).max()]) # search limits
dX_limits = X_limits / 40 # maximum step size from a point
X = np.random.uniform(X_limits[0], X_limits[1], (n_points, len(input_data_labels)))
Y = mlp_reg_model_LNGC5mol.predict(X)
X_max = 0 # inputs where maximum dY/dX is observed
dY_dX_max = 0 # maximum dY/dX value
for i in range(n_points):
    dX = np.random.uniform(dX_limits[0], dX_limits[1], (n_neighbours, len(input_data_labels)))
    X1 = dX + X[i]
    Y1 = mlp_reg_model_LNGC5mol.predict(X1)
    dY = Y1 - Y[i]
    dX_norm2 = np.linalg.norm(dX, ord=2, axis=1)
    dY_dX = np.max(np.abs(dY/dX_norm2))
    if dY_dX > dY_dX_max:
        dY_dX_max = dY_dX
        X_max = X[i]

print("Empirical Lipschitz constant is", np.round(dY_dX_max, 3), "at X", np.round(X_max, 3))
sys.exit()

#           EXPLORE THE LIPSCHITZ GRAPHICALLY

combs = list(itertools.combinations(range(len(input_data_labels)), 2)) # all 2-element combinations for x y z visualisation
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=tuple(['x: ' + input_data_labels[i] + '; y: ' + input_data_labels[j] for (i, j) in combs])
)
colorscales = ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Twilight']
for idx, comb in enumerate(combs):
    comb_inv = [i for i in range(len(input_data_labels)) if i not in comb] #inputs outside of a combination
    X = np.random.uniform(X_limits[0], X_limits[1], (n_points, len(input_data_labels)))
    X[:,comb_inv[0]] = X_max[comb_inv[0]] # set fixed value from the max dY/dX point
    X[:,comb_inv[1]] = X_max[comb_inv[1]] # set fixed value from the max dY/dX point
    Y = mlp_reg_model_LNGC5mol.predict(X)
    dY_dX_values = []
    for i in range(n_points):
        dX = np.random.uniform(dX_limits[0], dX_limits[1], (n_neighbours, len(input_data_labels)))
        dX[:,comb_inv[0]] = 0 # set zero increment for fixed input from the max dY/dX point
        dX[:,comb_inv[1]] = 0 # set zero increment for fixed input from the max dY/dX point
        X1 = dX + X[i]
        Y1 = mlp_reg_model_LNGC5mol.predict(X1)
        dY = Y1 - Y[i]
        dX_norm2 = np.linalg.norm(dX, ord=2, axis=1)
        dY_dX_values.append(np.max(np.abs(dY/dX_norm2)))
    xi = np.linspace(X_limits[0], X_limits[1], 100)
    yi = np.linspace(X_limits[0], X_limits[1], 100)
    x, y = np.meshgrid(xi, yi)
    z = griddata((X[:,comb[0]], X[:,comb[1]]), dY_dX_values, (x, y), method='linear')
    fig.add_trace(go.Surface(z=z, x=x, y=y, colorscale=colorscales[idx], colorbar=dict(title='dY/dZ - '+str(idx+1), x=(idx-int(idx/3)*3+1)/3-0.05+int(idx/3)/20)), row=int(idx/3)+1, col=idx-int(idx/3)*3+1)

fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
fig.update_layout(title=dict(text='Distribution of local Lipschitz estimates around max dY/dX point'), autosize=True,
                  margin=dict(l=5, r=5, b=10, t=60),
)

# Animation
frames = []
for angle in range(0, 360, 5):  # 5-degree steps
    rad = np.radians(angle)
    camera = dict(
        eye=dict(x=2*np.cos(rad), y=2*np.sin(rad), z=1.25)
    )
    frames.append(go.Frame(layout={
            'scene': dict(camera=camera),
            'scene2': dict(camera=camera),
            'scene3': dict(camera=camera),
            'scene4': dict(camera=camera),
            'scene5': dict(camera=camera),
            'scene6': dict(camera=camera),
        }))
fig.frames = frames
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'buttons': [{
            'label': 'Rotate',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 1 , 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }],
)
fig.show()
