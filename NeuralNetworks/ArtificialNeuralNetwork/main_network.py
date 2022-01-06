import sys
import numpy as np
from matplotlib import pyplot as plt
from network import Network
from functions_network import data_loader, plot_figure, plot_scores

#python main_network.py 0
data_type = sys.argv[1]
# Se grafican los resultados para 2 tipos de datos: 
#   0: circulos
#   1: lunas

#Arquitectura de red 
# [input layer , hidden layers, ..., output layer]
model = [2, 4, 4, 3]   #lista con el número de neuronas en cada capa 
neurons_input_layer = model[0]
neurons_output_layer = model[-1]

# Hiperparámetros modelo
epochs = 500 #500
alpha = 0.01  # alpha tasa de aprendizaje para el entrenamiento de la red
eta = 0.1     # eta tasa de aprendizaje para descenso por gradiente
mini_batch_size = 15  # hiperparámetro descenso por gradiente estocástico

(X, y, X_train, X_test, y_train , y_test, train_data, test_data)= data_loader(data_type, neurons_input_layer, neurons_output_layer)

network = Network(model, alpha)
network.train(train_data, epochs, eta, test_data, SGD=True, mini_batch_size=mini_batch_size)

plot_figure(network, X, X_train, y_train, X_test, y_test, epochs) 

#plot_decision_boundary(X, y, network)

#plot_scores(network.scores)




