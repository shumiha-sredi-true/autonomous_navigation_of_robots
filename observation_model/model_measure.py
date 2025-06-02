import matplotlib.pyplot as plt
import numpy as np

rmse_guss = 0.5
me = 4
x = np.linspace(0, 12, 1201)
p_gauss = np.sqrt(2 * np.pi * rmse_guss ** 2) ** (-1) * np.exp(-0.5 * (x - me) ** 2 / rmse_guss ** 2)

plt.figure(1)
plt.plot(x, p_gauss)
plt.grid()

index_me = np.where(x == me)[0]
l = 0.5
p_exp = l * np.exp(-l * x)
p_exp = np.where(p_exp < p_exp[index_me], 0, p_exp)

plt.figure(2)
plt.plot(x, p_exp)
plt.grid()

p_delta_f = np.copy(x)
p_delta_f = np.where(p_delta_f < 11.7, 0, p_delta_f)
p_delta_f = np.where(p_delta_f > 1, 1, p_delta_f)

plt.figure(3)
plt.plot(x, p_delta_f)
plt.grid()

p_rand = np.max(x) ** (-1) * np.ones(len(x))
plt.figure(4)
plt.plot(x, p_rand)
plt.grid()

# p_sum = p_gauss + p_exp + p_delta_f + p_rand

w_hit = 0.5
w_short = 0.1
w_max = 0.1
w_rand = 0.3
p_w = np.array([p_gauss, p_exp, p_delta_f, p_rand])
z_w = np.array([w_hit, w_short, w_max, w_rand])

p_sum = z_w@p_w
plt.figure(5)
plt.plot(x, p_sum)
plt.grid()
plt.show()
