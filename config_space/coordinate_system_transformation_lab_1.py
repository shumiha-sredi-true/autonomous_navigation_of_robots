import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([1.0, 0.5, np.pi / 4])
xld = np.array([0.2, 0.0, np.pi])

scan = np.array(np.loadtxt("laserscan.dat"))
angle = np.linspace(-np.pi / 2, np.pi / 2, scan.shape[0])
x_target_sensor = np.vstack([scan, scan]) * np.array([np.cos(angle), np.sin(angle)])

plt.figure(1)
plt.title("Система координат лазера")
plt.plot(x_target_sensor[0, :], x_target_sensor[1, :], "-o")
plt.xlabel("X_s")
plt.ylabel("Y_s")
plt.grid()

x_target_local = (np.array([[np.cos(xld[2]), -np.sin(xld[2])],
                            [np.sin(xld[2]), np.cos(xld[2])]]) @ x_target_sensor
                  + xld[:2].reshape((2, 1)) @ np.ones((1, scan.shape[0])))

plt.figure(2)
plt.title("Локальная система координат")
plt.plot(x_target_local[0, :], x_target_local[1, :], "-o")
plt.xlabel("X_l")
plt.ylabel("Y_l")
plt.grid()

x_target_global = (np.array([[np.cos(x0[2]), -np.sin(x0[2])],
                             [np.sin(x0[2]), np.cos(x0[2])]]) @ x_target_local
                   + x0[:2].reshape((2, 1)) @ np.ones((1, scan.shape[0])))

x_laser_global = (np.array([[np.cos(x0[2]), -np.sin(x0[2])],
                             [np.sin(x0[2]), np.cos(x0[2])]]) @ x_target_local
                    + x0[:2].reshape((2, 1)) @ np.ones((1, scan.shape[0])))
plt.figure(3)
plt.title("Глобальная система координат")
plt.plot(x_target_global[0, :], x_target_global[1, :], "-o")
plt.plot(x0[0], x0[1], '-o')

plt.xlabel("X_g")
plt.ylabel("Y_g")
plt.grid()
plt.show()
