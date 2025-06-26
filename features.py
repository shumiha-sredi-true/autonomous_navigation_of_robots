import numpy as np
import matplotlib.pyplot as plt

# -----------Многомерное распределение-------------------------
# Многомерное распределение с MO x=1,y = 2 и заданной дисперсией
cov = np.array([[0.25, 0], [0, 0.25]])  # матрица дисперсий
pts = np.random.multivariate_normal([1, 2], cov, size=800)

# -----------Построение графиков-------------------------------
# plt.figure(figsize=(6, 6))  # размер изображения
# plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)  # точки в виде полых кругов

"plt.hist(x_u[:, 0], bins=30, density=True, alpha = 0.6)"
"density = True - нормировка плотности вероятности"

# 3D график
x = np.linspace(0.0001, 0.6, 1000)
y = np.linspace(0.0001, 1, 1000)
KK = x/(x+0.07)
plt.plot(x, KK)
plt.grid()
plt.xlabel("D_ksi, (м/с)^2")
plt.ylabel("K")
plt.show()

K = np.zeros((len(x), len(y)))
for k, y_k in enumerate(y):
    for i, x_i in enumerate(x):
        K[k, i] = x_i/(x_i+y_k)
X, Y = np.meshgrid(x, y)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Рисуем поверхность
surf = ax.plot_surface(X, Y, K, cmap='viridis')

# Настройки графика
ax.set_xlabel('Ось X')
ax.set_ylabel('Ось Y')
ax.set_zlabel('Ось Z')
ax.set_title('3D поверхность')
fig.colorbar(surf)  # Добавляем цветовую шкалу

plt.show()