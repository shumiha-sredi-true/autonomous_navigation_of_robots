import numpy as np
import matplotlib.pyplot as plt

np.random.seed(13)

# Константы
T = 180
dT = 0.1
N = int(T/dT)
v_k = np.array([0.5, 0.4])
omega_k = np.array([0.03, 0.01])
tetta_k = np.array([0.2, 0.2])

for i in range(N):
    G_k = np.array([
        [1, 0, (v_k[0]/omega_k[0])*(-np.cos(tetta_k[i])+np.cos(tetta_k[i] + omega_k[i]*dT))],
        [0, 1, (v_k[0] / omega_k[0]) * (-np.sin(tetta_k[i])+np.sin(tetta_k[i] + omega_k[i]*dT))],
        [0, 0, 1]
    ])
 