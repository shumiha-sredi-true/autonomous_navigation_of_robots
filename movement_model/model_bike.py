import numpy as np
import matplotlib.pyplot as plt

"""
Дано:
    d - диаметр колеса
    l - длина рамы
    alpha - поворот переднего колеса вокруг вертикальной
    -> заднее колесо не поворачивается
    -> конфигурационное пространство описывается тремя компонентами: x, y, tetta
    -> скорость v и угол alpha в течение времени движения считать постоянными

Требуется:
    -> сформулировать модель прогнозирования в интервале delta_t
    -> v и alpha подвержены гауссову шуму
"""
l = 0.165
d = 0.622
v = 3.5
s = 800
delta_t = 1
D_v = 0.2**2
D_alpha = np.radians(1)**2
t = np.arange(0, s/v, delta_t)
noise_v = np.random.normal(0, np.sqrt(D_v), len(t))
noise_alpha = np.random.normal(0, np.sqrt(D_alpha), len(t))
alpha = np.radians(0)

tetta_0 = np.radians(20)
x_0 = 1
y_0 = 2

x_t_true = np.zeros((len(t), 3))
x_t_noise = np.zeros((len(t), 3))

x_t_true[0] = np.array([x_0, y_0, tetta_0])
x_t_noise[0] = np.array([x_0, y_0, tetta_0])

x_t_prev_true = x_t_true[0]
x_t_prev_noise = x_t_noise[0]
for i in range(1, len(t)):
    x_t_true[i] = x_t_prev_true + np.array([v*delta_t*np.cos(x_t_prev_true[2]), v*delta_t*np.sin(x_t_prev_true[2]),
                                            delta_t*v*np.tan(alpha)/l])
    x_t_prev_true = x_t_true[i]

    x_t_noise[i] = x_t_prev_noise + np.array([(v + noise_v[i - 1])*delta_t*np.cos(x_t_prev_noise[2]),
                                              (v + noise_v[i - 1])*delta_t*np.sin(x_t_prev_noise[2]),
                                              (v + noise_v[i - 1])*delta_t*
                                              (np.tan(alpha) + noise_alpha[i-1]/np.cos(alpha)**2)/l
                                              ])
    x_t_prev_noise = x_t_noise[i]

    if i == 10:
        alpha = np.radians(10)
    if i == 14:
        alpha = 0

plt.plot(t, x_t_true[:, 0])
plt.plot(t, x_t_noise[:, 0])
plt.grid()
plt.show()



