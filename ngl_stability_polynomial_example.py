# -*- coding: utf-8 -*-
"""
Created on Fri May 30 12:39:49 2025

@author: Ivan Nemov
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Parameters
np.random.seed(42)
n = 16  # Degree of the polynomial (change this)
num_points = 24
noise_std = 0.25

# Generate data
x = np.linspace(-np.pi, np.pi, num_points)
y_true = np.sin(x)
y_noisy = y_true + np.random.normal(scale=noise_std, size=num_points)

# Fit polynomial
coeffs = np.polyfit(x, y_noisy, deg=n)
poly = np.poly1d(coeffs)

# Generate fit values
x_fit = np.linspace(-np.pi, np.pi, 500)
y_fit = poly(x_fit)

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(x, y_noisy, label='Noisy samples', color='red', alpha=0.6)
plt.plot(x, y_true, label='True sin(x)', linestyle='--', color='green')
plt.plot(x_fit, y_fit, label=f'Polynomial fit (degree={n})', color='blue')
plt.title(f'Polynomial Approximation of sin(x) with Noise (Degree = {n})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()