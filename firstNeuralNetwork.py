# -*- coding: utf-8 -*-
import numpy
import scipy.special

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


n = neuralNetwork(3, 3, 3, 0.35)
print("Before train")
print("Query network:",n.query([3.2,2.1,-0.2]))
#print(n.who)
#print(n.wih)
for i in range(1000):
    n.train([3.2,2.1,-0.2], [5.6,0.3,0])
    pass
print("After train")
print("Query network:",n.query([3.2,2.1,-0.2]))
#print(n.who)
#print(n.wih)