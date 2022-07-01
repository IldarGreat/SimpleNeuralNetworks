# -*- coding: utf-8 -*-
import numpy
import scipy.special
import matplotlib.pyplot as plt

class neuralNetwork:

    # Инициализируем тренировочную сеть, количеством узлов в каждом слое и коэффициентом обучения
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lrate = learning_rate
        # Создаем матрицу весовых коэффициенто для входного-скрытого слоя
        # Функция normal - возвращет матрицу с нормальным распределением, первый аргуемент - центр, второй - старнадртное отклонение, третий - размер
        self.wih = numpy.random.normal(0.0,pow(hidden_nodes,-0.5),(hidden_nodes,input_nodes))
        # Создаем матрицу весовых коэффициенто для скрытого-выходного слоя
        self.who = numpy.random.normal(0.0,pow(hidden_nodes,-0.5),(output_nodes,hidden_nodes))
        # Определение функции активации с помощью лямбда выражения
        self.activation_function = lambda x : scipy.special.expit(x)
        pass
    
    # Тренируем сеть
    # Состоит из двух фаз, первая - расчет выходных сигналов(делает query), вторая - обратное распространение ошибок
    def train(self,input_list,target_list):
        targets = numpy.array(target_list,ndmin=2).T
        inputs = numpy.array(input_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)
        self.who += self.lrate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lrate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    
    # Опрос сети
    def query(self,input_list):
        # Преобразование воходящего списка в двумерный массив
        inputs = numpy.array(input_list,ndmin=2).T
        # Входящие сигналы из скрытого узла
        hidden_inputs = numpy.dot(self.wih,inputs)
        # Выходящие сигналы из скрытого узла
        hidden_outputs = self.activation_function(hidden_inputs)
        # Входящие сигналы для выходного узла
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # Конечный выход
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass
    pass


# Создание объекта нейронной сети
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
network = neuralNetwork(input_nodes, hidden_nodes, output_nodes, 0.35)
data_train_file = open("mnist_dataset/mnist_train_100.csv",'r')
data_values = data_train_file.readlines()
data_train_file.close()

# Тренировка сети
for record in data_values:
    data_value = record.split(",")
    inputs = (numpy.asfarray(data_value[1:])/255.0*0.99)+0.01
    targets = numpy.zeros(output_nodes) + 0.1
    targets[int(data_value[0])] = 0.99
    network.train(inputs,targets)
    pass
    
# Тестирование сети
data_test_file = open("mnist_dataset/mnist_test_10.csv",'r')
test_values = data_test_file.readlines()
data_train_file.close()
    
test_value = test_values[3].split(',')
image_array = numpy.asfarray(test_value[1:]).reshape((28,28))
plt.imshow(image_array,cmap = 'Greys',interpolation = "nearest")
inputs_data = numpy.asfarray(test_value[1:])/255.0*0.99 + 0.1

print(network.query(inputs_data))