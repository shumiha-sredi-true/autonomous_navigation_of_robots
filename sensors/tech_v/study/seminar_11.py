import matplotlib.pyplot as plt
import numpy as np
import json

with open("vo_step_10_deg.json", "r", encoding="utf-8") as file:
    data = json.load(file)

angle_pitch = []
time_0 = data[0]["time"]
time_exp = []
x = []
y = []


angle_true = np.linspace(-90,90, 19)
x_true = np.cos(np.radians(angle_true))
y_true = np.sin(np.radians(angle_true))

for i in range(len(data)):
    time_exp.append(data[i]["time"] - time_0)
    angle_pitch.append(data[i]["pitch_deg"])

    x.append(np.cos(np.radians(angle_pitch[-1])))
    y.append(np.sin(np.radians(angle_pitch[-1])))


plt.figure(1)
plt.scatter(np.array(time_exp), angle_pitch, label = "Курс", color = "red")
plt.xlabel("Время, секунды")
plt.ylabel("Угол, градусы")
plt.axhline(-10)
plt.axhline(-20)
plt.axhline(-30)
plt.axhline(-40)
plt.axhline(-50)
plt.axhline(-60)
plt.axhline(-70)
plt.axhline(-80)
plt.axhline(-90)
plt.axhline(10)
plt.axhline(20)
plt.axhline(30)
plt.axhline(40)
plt.axhline(50)
plt.axhline(60)
plt.axhline(70)
plt.axhline(80)
plt.axhline(90)
plt.legend()
plt.grid()


plt.figure(2)

plt.scatter(y,x , label = "Курс", color = "red")
plt.scatter(y_true,x_true , label = "Курс", color = "blue")
plt.grid()

plt.show()
