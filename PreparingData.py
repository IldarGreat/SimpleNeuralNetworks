# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 19:08:24 2022

@author: ildar
"""
import numpy as np
import matplotlib.pyplot as plt

# Открываем тренировочный файл
data_file = open("mnist_dataset/mnist_train_100.csv",'r')
# Считываем абсолютно все строки
data_list = data_file.readlines()
data_file.close()

# Тут можно выбрать любой экземпляр(0-99) из этого файла и построить его график
all_values = data_list[4].split(",")
# Создаем матрицу 28x28 из элементов массива(за исключение первого элемента - он маркер)
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array,cmap = 'Greys',interpolation= 'None')

# Подготовка входных данных
prepare_input = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01

# Подготовка целевых данных
# 0.99 будет означать целевой результат
nodes = 10
targets = np.zeros(nodes) + 0.1
targets[int(all_values[0])] = 0.99
print(targets)