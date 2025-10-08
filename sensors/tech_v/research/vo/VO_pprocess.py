import matplotlib.pyplot as plt
import numpy as np
import json

with open("vo_step_10_deg.json", "r", encoding="utf-8") as file:
    data = json.load(file)

time_min = 2
time_n = time_min * 60

angle_yaw = []
angle_roll = []
angle_pitch = []
for i in range(len(data)):
    angle_yaw.append(data[i]["yaw_deg"])
    angle_roll.append(data[i]["roll_deg"])
    angle_pitch.append(data[i]["pitch_deg"])

dt = time_n/len(angle_yaw)

plt.scatter(np.arange(0, len(angle_yaw), 1)*dt, angle_yaw, label = "Курс")
plt.scatter(np.arange(0, len(angle_yaw), 1)*dt, angle_roll, label = "Roll")
plt.scatter(np.arange(0, len(angle_yaw), 1)*dt, angle_pitch, label = "Pitch")
plt.axhline(-10)
plt.axhline(-20)
plt.axhline(-30)
plt.axhline(-40)
plt.axhline(-50)
plt.axhline(-70)
plt.axhline(-80)
plt.axhline(-90)
plt.axhline(10)
plt.axhline(20)
plt.axhline(30)
plt.axhline(40)
plt.axhline(50)
plt.axhline(70)
plt.axhline(80)
plt.axhline(90)
plt.legend()
plt.grid()
plt.show()
