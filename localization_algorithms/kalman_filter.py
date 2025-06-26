import numpy as np
import matplotlib.pyplot as plt
np.random.seed(13)
# Константы
T = 180
dT = 0.1
a = 0.1
D_ksi = 0.25 ** 2  # формирующий шум ускорения
# D_ksi = np.array([[0.07**2, 0],
#                   [0, 0.07 ** 2]]) # формирующий шум ускорения
D_n = 0.07 ** 2  # радар TI AWR1843 точность по скорости ±0.03–0.1 м/с

# Время разгона составляет: 10 секунд
ACC = int(10/dT)
u_1 = np.array([a for _ in range(int(ACC))])

# Время переходного процесса при отключении ускорения: 5 секунд
P1 = int(5/dT)
u_2 = np.array([a - a * (_ / (P1 + 1)) for _ in range(1, P1 + 1)])

# Время равномерного движения: 50 секунд
R1 = int(50/dT)
u_3 = np.array([0.0 for _ in range(R1)])

# Время переходного процесса при включении ускорения: 5 секунд
P2 = int(5/dT)
u_4 = np.array([- a * (_ / (P2 + 1)) for _ in range(1, P2 + 1)])

# Время смены направления движения: 20 секунд (движение в другую сторону)
ACC_rev = int(21/dT)
u_5 = np.array([-(a - a * (_ / ACC_rev)) for _ in range(1, ACC_rev + 1)])

# Время равномерного движения: 30 секунд
R2 = int(89/dT)
u_6 = np.array([0 for _ in range(R2)])


N = ACC+P1+P2+R1+R2+ACC_rev
u = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([u_1, u_2]), u_3]), u_4]), u_5]), u_6])

"""
    Линейная модель: автомобиль с управляемым ускорением
    Вектор состояния: x = np.array([x , v])
    Вектор наблюдений: z = v - наблюдения  радара TI AWR1843
    
    Предполагаем, что модель изменения информационных параметров - марковская
    x_k = x_k-1 + v*dT + 0.5 *a* dT^2
    v_k = v_k-1 + a*dT
    
    x = F*x + B*u + G*ksi
"""

F = np.array([[1, dT],
              [0, 1]])
B = np.array([dT ** 2 / 2, dT])
H = np.array([0, 1])
G = np.array([0, dT])

# Моделирование движения в виде марковского процесса
x_true = np.zeros((N, 2))
z = np.zeros(N)
x_prev = np.array([0, 0])

for i in range(N):
    # x_true[i] = F @ x_prev + B * u[i] + G * np.random.normal(0, np.sqrt(D_ksi))
    x_true[i] = F @ x_prev + B * u[i]
    z[i] = H @ x_true[i] + np.random.normal(0, np.sqrt(D_n))
    x_prev = x_true[i]

z_prev = 0
z_k = np.zeros(len(z))
for i, v in enumerate(z):
    z_k[i] = z_prev + z[i] * dT
    z_prev = z_k[i]

fig1, ax = plt.subplots(1, 2, figsize=(10, 8))
ax[0].set_title("Скорость")
ax[0].plot(np.arange(0, N * dT, dT), z, label="наблюдения")
ax[0].plot(np.arange(0, N * dT, dT), x_true[:, 1], label="истина")
ax[0].set_xlabel("Время, сек.")
ax[0].set_ylabel("Скорость, м/c")
ax[0].legend()
ax[0].grid()

ax[1].set_title("Координата")
ax[1].plot(np.arange(0, N * dT, dT), z_k, label="наблюдения")
ax[1].plot(np.arange(0, N * dT, dT), x_true[:, 0], label="истина")
ax[1].set_xlabel("Время, сек.")
ax[1].set_ylabel("Координата x, м")
ax[1].legend()
ax[1].grid()


# Фильтр Калмана
x_prev = np.array([0, 0])
D_prev = np.array([[0, 0],
                   [0, 0]])
x_est = np.zeros((N, 2))
D_est = np.zeros((N, 2, 2))

for i in range(N):
    # Шаг экстраполяции
    x_ext = F @ x_prev + B * u[i]
    D_ext = F @ D_prev @ F.T + G * D_ksi * G.T

    # Шаг коррекции
    K = D_ext @ H.T * (H.T @ D_ext @ H + D_n) ** (-1)
    D_est[i] = (np.eye(2) - K @ H) @ D_ext
    x_est[i] = x_prev + K * (z[i] - H @ x_ext)

    x_prev = x_est[i]
    D_prev = D_est[i]

print("Среднее значение динамической ошибки ", np.mean(x_true[:, 1] - x_est[:, 1]))
print("СКО ошибки ", np.std(x_true[:, 1] - x_est[:, 1]))
print("good")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].set_title("Оценка скорости движения")
axes[0, 0].plot(np.arange(0, N * dT, dT), z, label="наблюдения")
axes[0, 0].plot(np.arange(0, N * dT, dT), x_true[:, 1], label="истина")
axes[0, 0].plot(np.arange(0, N * dT, dT), x_est[:, 1], label="оценка")
axes[0, 0].legend()
axes[0, 0].set_xlabel("Время, сек.")
axes[0, 0].set_ylabel("Скорость, м/c")
axes[0, 0].grid()

axes[0, 1].set_title("СКО ошибки оценки")
axes[0, 1].plot(np.arange(0, N * dT, dT), D_est[:, 1, 1], label="СКО скорости")
axes[0, 1].legend()
axes[0, 1].grid()

axes[1, 0].set_title("Ошибка оценки")
axes[1, 0].plot(np.arange(0, N * dT, dT), x_true[:, 1] - x_est[:, 1], color = "green", label="дин. ошибка")
axes[1, 0].legend()
axes[1, 0].set_ylim(-0.4, 0.4)
# axes[1, 0].grid()

# axes[1, 1].set_title("Динамическая ошибка")
# axes[1, 1].plot(np.arange(0, N * dT, dT), x_true[:, 1] - x_est[:, 1] - np.mean(x_true[:, 1] - x_est[:, 1]),
#                 label="флукт. ошибка")
# axes[1, 1].grid()
plt.tight_layout()  # Чтобы подписи не накладывались
plt.show()
