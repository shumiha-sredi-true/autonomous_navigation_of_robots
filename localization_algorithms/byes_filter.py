import numpy as np
import matplotlib.pyplot as plt

"""
Демонстрация работы фильтра Байеса

Примечание:
1) Для описания состояния робота был выбран нормальный закон распределения
2) Модель движения: на основе одометрии
3) Модель наблюдений: нормальный распределение оценки СШП с ненулевым МО и СКО 10 см.
"""
check_flag = False  # проверка модели движения
N = 1000  # количество рассматриваемых состояний в фильтре
x = np.linspace(0, 4, N)
y = np.linspace(2, 8, N)


def odometry_model_move(x_t, u_t, alfa, N):
    """
    Функция определяет следующее положение робота исходя из данных одометрии
    и информации о предыдущем положении робота (выдает массив вероятных положений робота)

    :param x_t: [x, y, tetta]
    :param u_t: [tetta_rot_1, tetta_rot_2, tetta_trans]
    :param alfa: [alfa_1, alfa_2, alfa_3, alfa_4]
    :param N: int
    :return:
    """
    x_t_ext = np.zeros((N, 3))
    for i in range(x_t_ext.shape[0]):
        tetta_rot_1_est = u_t[0] - np.random.normal(0, np.sqrt(alfa[0] * u_t[0] ** 2 + alfa[1] * u_t[2] ** 2))
        tetta_trans_est = u_t[2] - np.random.normal(0,
                                                    np.sqrt(
                                                        alfa[2] * u_t[2] ** 2 + alfa[3] * u_t[0] ** 2 + alfa[3] * u_t[
                                                            1] ** 2))
        tetta_rot_2_est = u_t[1] - np.random.normal(0, np.sqrt(alfa[0] * u_t[1] ** 2 + alfa[1] * u_t[2] ** 2))

        x_next = np.zeros(3)
        x_next[0] = x_t[0] + tetta_trans_est * np.cos(x_t[2] + tetta_rot_1_est)
        # print(tetta_trans_est * np.cos(x_t[2] + tetta_rot_1_est))
        # print(str(x_t[0]) + " --> " + str(x_next[0]))
        x_next[1] = x_t[1] + tetta_trans_est * np.sin(x_t[2] + tetta_rot_1_est)
        x_next[2] = x_t[2] + tetta_rot_1_est + tetta_rot_2_est
        x_t_ext[i] = x_next

    return x_t_ext


if check_flag:
    x_t = np.array([2, 4, 0])
    u_t = np.array([np.pi / 3, 0, 1])
    alfa = np.array([0.01, 0.1, 0.001, 0.001])

    x_u = odometry_model_move(x_t, u_t, alfa, 1000)

    plt.title("Карта XY")
    plt.scatter(x_u[:, 0], x_u[:, 1], color="blue", label="Вероятностное поле состояний робота x_k+1")
    plt.scatter(x_t[0], x_t[1], color="red", label="Начальное положение x_k")
    plt.legend()
    plt.grid()
    plt.show()

    x_mean = np.mean(x_u[:, 0])
    x_std = np.std(x_u[:, 0])
    x = np.arange(0, 4, 0.004)
    x_normal = np.sqrt(2 * np.pi * x_std ** 2) ** (-1) * np.exp(-((x - x_mean) ** 2) / (2 * x_std ** 2))

    y_mean = np.mean(x_u[:, 1])
    y_std = np.std(x_u[:, 1])
    y = np.arange(0, 8, 0.008)
    y_normal = np.sqrt(2 * np.pi * y_std ** 2) ** (-1) * np.exp(-((y - y_mean) ** 2) / (2 * y_std ** 2))

    plt.plot(x, x_normal, color="blue")
    plt.plot(y, y_normal, color="green")
    plt.hist(x_u[:, 0], bins=30, density=True, alpha=0.6)
    plt.hist(x_u[:, 1], bins=30, density=True, alpha=0.6)
    plt.show()

# Определение сигналов управления
x_t = np.array([2, 4, 0])  # Начальное положение
x_target = np.array([3, 5, 0])  # Конечное положение
norm = x_target - x_t  # Норма векторов
ang = np.arctan(norm[1] / norm[0])  # Угол между векторами
u_t = np.array([np.arctan(norm[1] / norm[0]), 0, np.sqrt(norm[0] + norm[1])])  # Сигнал управления
alfa = np.array([0.01, 0.1, 0.001, 0.001])  # Неопределенность модели движения

# Априорное распределение координат в начальном положении
x0 = np.vstack([x, y]).T  # Объединение в двумерный массив
x0_rmse = np.array([0.05, 0.05])  # СКО координат
x0_me = np.array([x_t[0], x_t[1]])  # МО координат
bel_x = np.sqrt(2 * np.pi * x0_rmse ** 2) ** (-1) * np.exp(-((x0 - x0_me) ** 2) / (2 * x0_rmse ** 2))  # Bel_x_0

# Шаг предсказания
p_sum = 0
for i in range(N):
    x_u = odometry_model_move(np.array([x0[i][0], x0[i][1], 0]), u_t, alfa, N)
    x_u_me = np.array([np.mean(x_u[:, 0]), np.mean(x_u[:, 1])])
    x_u_rmse = np.array([np.std(x_u[:, 0]), np.std(x_u[:, 1])])
    p_xt_ut = np.sqrt(2 * np.pi * x_u_rmse ** 2) ** (-1) * np.exp(-((x0 - x_u_me) ** 2) / (2 * x_u_rmse ** 2))
    p_sum = p_sum + p_xt_ut * bel_x[i]

# Коррекция
# Наблюдения: СШП
rmse_uwb = np.array([0.1, 0.1])
me_uwb = np.array([x_target[0] + 0.03, x_target[1] + 0.03])
p_z = np.sqrt(2 * np.pi * rmse_uwb ** 2) ** (-1) * np.exp(-((x0 - me_uwb) ** 2) / (2 * rmse_uwb ** 2))
p = p_z * p_sum

fig, axes = plt.subplots(1, 2, figsize=(10, 8))
axes[0].set_title("X")
axes[0].plot(x, bel_x[:, 0] / np.max(bel_x[:, 0]), color="green", label="априорная информация")
axes[0].plot(x, p_sum[:, 0] / np.max(p_sum[:, 0]), color="blue", label="экстраполяция")
axes[0].plot(x, p_z[:, 0] / np.max(p_z[:, 0]), color="red", label="наблюдения")
axes[0].plot(x, p[:, 0] / np.max(p[:, 0]), color="black", label="коррекция")
axes[0].legend()
axes[0].grid()

axes[1].set_title("Y")
axes[1].plot(y, bel_x[:, 1] / np.max(bel_x[:, 1]), color="green", label="априорная информация")
axes[1].plot(y, p_sum[:, 1] / np.max(p_sum[:, 1]), color="blue", label="экстраполяция")
axes[1].plot(y, p_z[:, 1] / np.max(p_z[:, 1]), color="red", label="наблюдения")
axes[1].plot(y, p[:, 1] / np.max(p[:, 1]), color="black", label="коррекция")
axes[1].legend()
axes[1].grid()

# Сэмплинг (закон распределения ----> массив координат)
cov_bel_0 = np.array([[x0_rmse[0] ** 2, 0], [0, x0_rmse[1] ** 2]])
pts = np.random.multivariate_normal(x0_me, cov_bel_0, size=N)

cov_p_z = np.array([[rmse_uwb[0] ** 2, 0], [0, rmse_uwb[1] ** 2]])
pts_p_z = np.random.multivariate_normal(me_uwb, cov_p_z, size=N)

cov_p_pu = np.array([[0.2 ** 2, 0], [0, 0.2 ** 2]])
pts_p_pu = np.random.multivariate_normal(me_uwb - np.array([0.01, 0.05]), cov_p_pu, size=N)

plt.figure(figsize=(6, 6))  # размер изображения
# plt.scatter(x_t[0], x_t[1], color = "red", s=100)
# plt.scatter(x_target[0], x_target[1], color = "red", s=100)
# plt.quiver(x_t[0], x_t[1], norm[0], norm[1], angles='xy', scale_units='xy', scale=1, color='black')
plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5, label="Начальное положение")
plt.plot(pts_p_z[:, 0], pts_p_z[:, 1], '.', alpha=0.5, label="Наблюдение")
plt.plot(pts_p_pu[:, 0], pts_p_pu[:, 1], '.', alpha=0.5, label="Экстраполяция")
plt.legend()
plt.grid()
plt.show()
