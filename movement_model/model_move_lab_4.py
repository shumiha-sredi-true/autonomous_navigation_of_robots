import numpy as np
from matplotlib import pyplot as plt
import timeit


# I. Сэмплинг
def sample_normal_distribution(mean_, d, N):
    sum_normal = np.zeros(N)
    sigma_ = np.sqrt(d)
    left_ = mean_ - sigma_
    right_ = mean_ + sigma_
    for i in range(1, 13):
        sum_normal = sum_normal + np.random.uniform(left_, right_, size=N)
    # plt.figure(1)
    # plt.hist(0.5 * sum_normal, bins=100)
    return 0.5 * sum_normal


def boks_muller(mean_, sigma_, N):
    # u1 и u2 [0, 1]
    u1 = np.random.uniform(0, 1, size=N)
    u2 = np.random.uniform(0, 1, size=N)
    x = np.cos(2 * np.pi * u1) * np.sqrt(-2 * np.log(u2))
    x_end = mean_ + x * sigma_
    # plt.figure(2)
    # plt.hist(x_end, bins=100)
    return x_end


check_f = False
if check_f:
    # Проверка
    sample_n = sample_normal_distribution(0, 5, 10000)
    print(f"CКО суммы 12 сэмплов: {np.std(sample_n)}")

    sample_boks = boks_muller(0, 5, 10000)
    print(f"CКО сэмплов по формлуе Бокса-Мюллера: {np.std(sample_boks)}")

    sample_f = np.random.normal(0, 5, 10000)
    print(f"CКО сэмплов по встроенной функции: {np.std(sample_f)}\n")

    # Подсчет времени
    time_sample_n = timeit.timeit(lambda: sample_normal_distribution(0, 5, 100), number=100)
    print(f"Время sample_normal_distribution(): {time_sample_n}")
    time_sample_boks = timeit.timeit(lambda: boks_muller(0, 5, 100), number=100)
    print(f"Время boks_muller(): {time_sample_boks}")

    time_sample_f = timeit.timeit(lambda: np.random.normal(0, 5, 100), number=100)
    print(f"Время np.random.normal(): {time_sample_f}")

    plt.figure(1)
    plt.hist(sample_n, bins=100)
    plt.hist(sample_boks, bins=100)
    plt.hist(sample_f, bins=100)
    plt.grid()
    plt.show()


# II. Модель процесса на основе данных одометрии
def odometry_model(x_t, u_t, alfa):
    """
    Функция определяет следующее положение робота исходя из данных одометрии
    и информации о предыдущем положении робота

    :param x_t: [x, y, tetta]
    :param u_t: [tetta_rot_1, tetta_rot_2, tetta_trans]
    :param alfa: [alfa_1, alfa_2, alfa_3, alfa_4]
    :return:
    """

    tetta_rot_1_est = u_t[0] - np.random.normal(0, np.sqrt(alfa[0] * u_t[0] ** 2 + alfa[1] * u_t[2] ** 2))
    tetta_trans_est = u_t[2] - np.random.normal(0,
                                                np.sqrt(alfa[2] * u_t[2] ** 2 + alfa[3] * u_t[0] ** 2 + alfa[3] * u_t[
                                                    1] ** 2))
    tetta_rot_2_est = u_t[1] - np.random.normal(0, np.sqrt(alfa[0] * u_t[1] ** 2 + alfa[1] * u_t[2] ** 2))

    x_next = np.zeros(3)
    x_next[0] = x_t[0] + tetta_trans_est * np.cos(x_t[2] + tetta_rot_1_est)
    # print(tetta_trans_est * np.cos(x_t[2] + tetta_rot_1_est))
    # print(str(x_t[0]) + " --> " + str(x_next[0]))
    x_next[1] = x_t[1] + tetta_trans_est * np.sin(x_t[2] + tetta_rot_1_est)
    x_next[2] = x_t[2] + tetta_rot_1_est + tetta_rot_2_est

    return x_next


x_t = np.array([2, 4, 0])
u_t = np.array([np.pi / 2, 0, 1])
alfa = np.array([0.01, 0.1, 0.001, 0.001])
x_t_arr = np.zeros((5000, 3))
for i in range(5000):
    x_t_arr[i] = odometry_model(x_t, u_t, alfa)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 строки, 2 столбца
axes[0, 0].set_title("Карта XY")
axes[0, 0].scatter(x_t_arr[:, 0], x_t_arr[:, 1], color="blue", label = "Вероятностное поле состояний робота x_k+1")
axes[0, 0].scatter(x_t[0], x_t[1], color="red", label = "Начальное положение x_k")
axes[0, 0].legend()
axes[0, 0].grid()

axes[0, 1].set_title("Графики ординат")
axes[0, 1].plot(x_t_arr[:, 0], label = "X")
axes[0, 1].plot(x_t_arr[:, 1], label = "Y")
axes[0, 1].legend()
axes[0, 1].grid()

axes[1, 0].set_title("Значения угла")
axes[1, 0].plot(x_t_arr[:, 2], label = "Угол")
axes[1, 0].legend()
axes[1, 0].grid()

plt.tight_layout()
plt.show()
