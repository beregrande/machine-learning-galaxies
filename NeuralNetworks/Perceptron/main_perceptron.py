import sys
from functions_perceptron import data_loader, plot_figure, plot_scores
from perceptron import Perceptron

#Se pueden graficar los resultados para 3 tipos de datos: 
# 0: linealmente separables 2 clases
# 1: no linealmente separables
# 2: linealmente separables 3 clases
data_type = int(sys.argv[1])

#Se cargan los datos y se divide el conjunto de datos en entrenamiento y prueba
(X, y, X_train, y_train, X_test, y_test) = data_loader(data_type)

#Hiperparámetros modelos
epochs = 10
alpha = 0.01 # tasa de aprendizaje para entrenamiento
           
#Se crea el modelo perceptrón y se entrena el modelo
perceptron = Perceptron(alpha)
perceptron.train(X_train, y_train, epochs, test_data= (X_test, y_test))

# Grafica fronteras de decisión en diferentes observaciones 
folder = 'C:/Users/bereg/OneDrive/Escritorio/ProyectoTesis/output/'
plot_figure(perceptron, data_type, X, X_train, y_train, X_test, y_test, anim=False, out_folder=folder)
