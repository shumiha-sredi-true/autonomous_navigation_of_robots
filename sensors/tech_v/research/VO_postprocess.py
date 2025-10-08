import matplotlib.pyplot as plt
import numpy as np
import json

with open("yaw_data_dynamics.json", "r", encoding="utf-8") as file:
    data = json.load(file)

angle_yaw = []
angle_roll = []
angle_pitch = []
for i in range(len(data)):
    angle_yaw.append(data[i]["yaw_deg"])
    angle_roll.append(data[i]["roll_deg"])
    angle_pitch.append(data[i]["pitch_deg"])

time_n = 5 * 60
dt = time_n/len(angle_yaw)

# angle_roll = np.array(angle_roll)
# angle_pitch = np.array(angle_pitch)
angle_yaw = np.array(angle_yaw)

plt.scatter(np.arange(0, len(angle_yaw), 1)*dt, angle_yaw, label = "Курс")
plt.scatter(np.arange(0, len(angle_yaw), 1)*dt, angle_roll, label = "Roll")
plt.scatter(np.arange(0, len(angle_yaw), 1)*dt, angle_pitch, label = "Pitch")
# plt.axvline(3*60, color = "red")
# plt.axvline(3*60 + 15, color = "red")
plt.legend()
plt.grid()
plt.show()
