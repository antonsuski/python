import random
import numpy as np
import matplotlib.pyplot as plt

class vec2:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

def proc_avg_predict(arr):
    return round(sum(arr) / len(arr), 1)

def proc_slope(data, avg_predict_x , avg_predict_y):
    numerator = 0
    denomerator = 0
    for i in data:
        numerator += (i.x - avg_predict_x) * (i.y - avg_predict_y)
        denomerator += pow(i.x - avg_predict_x, 2)
    return numerator / denomerator

def proc_free_member(avg_predict_x, avg_predict_y, slope):
    return avg_predict_y - (slope * avg_predict_x)

def rss(data):
    x_pred = proc_avg_predict([i.x for i in data])
    y_pred = proc_avg_predict([i.y for i in data])
    print(x_pred, y_pred)

    slope_b1 = proc_slope(data, x_pred, y_pred);
    free_member_b0 = proc_free_member(x_pred, y_pred, slope_b1);
    print("slope b1:", slope_b1, " free member b0:", free_member_b0)

    acc = 0
    for i in data:
         acc += pow(i.y - free_member_b0 - (slope_b1 * i.x), 2)
    print("rss:", acc)
    return acc, free_member_b0, slope_b1

data = []

for i in range(10):
        x = round(random.uniform(0, 10), 1)
        y = round(random.uniform(0, 10), 1)
        data.append(vec2(x,y));

for point in data:
     print(point.x, point.y)

# Извлекаем координаты точек для графика
x_values = [point.x for point in data]
y_values = [point.y for point in data]

# Отображаем точки на графике
# plt.scatter(x_values, y_values, color='red', label='Points')

# # Добавляем заголовок и подписи осей
# plt.title('[points]')
# plt.xlabel('X')
# plt.ylabel('Y')

# # Включаем легенду и сетку
# plt.legend()
# plt.grid(True)

# Показываем график
# plt.show()

rssres, b0, b1 = rss(data)
