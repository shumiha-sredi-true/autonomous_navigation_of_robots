import numpy as np
import matplotlib.pyplot as plt

# Время наблюдения и статистические параметры
T_obs = 1e-3
sigma_a = 0.06
sigma_n = 1
alpha = T_obs

# Период дальномерного кода и длительность чипа
T_c = 1e-3
T_chip = T_c / 511

# Частота дискретизации и промежуточная частота
f_d = 33 * 1e+6
f_0 = 8 * 1e+6

N = int(T_obs * f_d)
T_d = 1 / f_d

q_e = (alpha*sigma_a**2)/(2*sigma_n**2*T_d)
h = 1
he = np.log((1 + q_e) * h) * ((alpha * sigma_n**2) / (2 * T_d))

q_cn = 48
tau_symbol = 1 / 511 * 1e-3

# Задание значений сетки 1022*9
tau_delay = np.linspace(0, 1, 1022) * 1e-3
f_dop = np.linspace(-2, 2, 9) * 1e+3
time_y = np.arange(0, 1e-3, T_d)
time_G_t = np.arange(0, 1e-3, tau_symbol)

fig1, axes = plt.subplots(2, 1, figsize=(10, 8))  # 2 строки, 2 столбца

# Дальномерный код G(t)
G_t = np.array(list(map(int, open("DK_LxOF.txt", "r").readlines())))
axes[0].set_title("Дальномерный код")
axes[0].plot(np.arange(0, 1e-3, tau_symbol) / 1e-3, G_t)
axes[0].set_xlabel("Время, мс")
axes[0].set_ylabel("Вольты")
axes[0].grid()

# Наблюдаемая реализация
Y_t = list(map(int, open("Yvar11.txt", "r").readlines()))
axes[1].plot(time_y / 1e-3, Y_t)
axes[1].plot(time_G_t / 1e-3, G_t, color="red")
axes[1].set_title('Наблюдаемая реализация')
axes[1].set_xlabel("Время, мс")
axes[1].set_ylabel("Вольты")
axes[1].grid()
plt.tight_layout()  # Чтобы подписи не накладывались

X_2 = np.zeros((len(tau_delay), len(f_dop)))
for i, t in enumerate(tau_delay):
    for j, fd in enumerate(f_dop):
        Nbyte = ((np.arange(N) * T_d + t) / T_chip % 511).astype(int)
        I = np.sum(Y_t * G_t[Nbyte] * np.cos(2 * np.pi * (f_0 + fd) * time_y))
        Q = np.sum(Y_t * G_t[Nbyte] * np.sin(2 * np.pi * (f_0 + fd) * time_y))

        X_2[i, j] = np.sum(I ** 2 + Q ** 2)

# Поиск максимума и оценка вероятности ложной тревоги
# Находим максимальное значение корреляции и его индексы
Mmax = np.max(X_2)
i_max, j_max = np.unravel_index(np.argmax(X_2), X_2.shape)

# Вычисляем задержку и доплеровскую частоту для максимального корреляционного отклика
tau_max = tau_delay[i_max]
fd_max = f_dop[j_max]

# Вычисляем вероятность ложной тревоги
Pfa = np.sum(X_2 >= he) / X_2.size

# Вывод результатов
print(f'Максимальное значение корреляции: {Mmax:.2f}')
print(f'Задержка сигнала: {tau_max * 1e3:.3f} мс')
print(f'Доплеровская частота: {fd_max:.1f} Гц')
print(f'Количество ячеек, где превышен порог: {np.sum(X_2 >= he)}')
print(f'Вероятность ложной тревоги: {Pfa:.4f}')


fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
F_dop, T_delay = np.meshgrid(f_dop, tau_delay)
ax.plot_surface(F_dop, T_delay, X_2, cmap='viridis')
ax.set_xlabel('F_dop')
ax.set_ylabel('T_delay')
ax.set_zlabel('X')
plt.show()

"""import numpy as np
import matplotlib.pyplot as plt

f_0 = 8 * 1e+6
T = 1e-3
f_d = 33 * 1e+6
T_d = 1 / f_d
sigma_a = 0.06
sigma_n = 1
q_cn = 48
tau_symbol = 1 / 511 * 1e-3

# Задание значений сетки 1022*9
tau_delay = np.linspace(0, 1, 1022) * 1e-3
f_dop = np.linspace(-2, 2, 2) * 1e+3
time_y = np.arange(0, 1e-3, T_d)
time_G_t = np.arange(0, 1e-3, tau_symbol)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 строки, 2 столбца

# Дальномерный код G(t)
G_t = list(map(int, open("DK_LxOF.txt", "r").readlines()))
axes[0, 0].set_title("Дальномерный код")
axes[0, 0].plot(np.arange(0, 1e-3, tau_symbol) / 1e-3, G_t)
axes[0, 0].set_xlabel("Время, мс")
axes[0, 0].set_ylabel("Вольты")
axes[0, 0].grid()

# Наблюдаемая реализация
Y_t = list(map(int, open("Yvar11.txt", "r").readlines()))
axes[0, 1].plot(time_y / 1e-3, Y_t)
axes[0, 1].plot(time_G_t / 1e-3, G_t, color="red")
axes[0, 1].set_title('Наблюдаемая реализация')
axes[0, 1].set_xlabel("Время, мс")
axes[0, 1].set_ylabel("Вольты")
axes[0, 1].grid()
plt.tight_layout()  # Чтобы подписи не накладывались

G_t_delay = np.array([[G_t[i], G_t[i]] for i in range(len(G_t))]).reshape(2 * len(G_t))
X = np.zeros((len(f_dop), len(tau_delay)))
for i in range(len(f_dop)):
    print("Частота Доплера = ", f_dop[i])
    G_i = np.copy(G_t_delay)
    for j in range(len(tau_delay)):
        print("Задержка = ", tau_delay[j])
        if j != 0:
            last_el = G_i[-1]
            G_i = np.insert(G_i, 0, last_el)[:-1]
        I = 0
        Q = 0
        t = 1
        print("j = ", j, "i = ", i)
        for k, t_d in enumerate(time_y):
            if t_d > tau_delay[1] * t:
                t += 1
            I = I + Y_t[k] * G_i[t - 1] * np.cos(2 * np.pi * (f_0 + f_dop[i]) * t_d)
            Q = Q + Y_t[k] * G_i[t - 1] * np.sin(2 * np.pi * (f_0 + f_dop[i]) * t_d)

        X[i, j] = np.sqrt(I ** 2 + Q ** 2)

ax = fig.add_subplot(111, projection='3d')
T_delay, F_dop = np.meshgrid(tau_delay,f_dop)
ax.plot_surface(T_delay, F_dop, X, cmap='viridis')
ax.set_xlabel('F_dop')
ax.set_ylabel('T_delay')
ax.set_zlabel('X')
plt.show()
"""
