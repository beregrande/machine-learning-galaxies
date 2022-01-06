import numpy as np
from numpy.core.fromnumeric import size 
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.colors as colors

def data_loader(data_type,neurons_input_layer, neurons_output_layer):
    '''
    Carga los datos
    '''
    if data_type=="0": 
        X, y = datasets.make_circles(200, noise = .1, factor = .4)    
    elif data_type=="1": 
        X, y = datasets.make_moons(200, noise=0.20)
    elif data_type=="2": 
        X, y = datasets.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.2, random_state=697)
    else:
        X, y = datasets.make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=1.2, random_state=697)


    X_train, X_test, y_train , y_test = train_test_split(X, y, test_size =0.2)

    training_inputs = [np.reshape(x, (neurons_input_layer, 1)) for x in X_train]
    training_results = [vectorized_result(y, neurons_output_layer) for y in y_train]
    train_data = list(zip(training_inputs, training_results))
    test_inputs = [np.reshape(x, (neurons_input_layer, 1)) for x in X_test]
    test_data = list(zip(test_inputs, y_test))

    return (X, y, X_train, X_test, y_train , y_test, train_data, test_data)

def plot_decision_boundary(boundary, X, model, ax=None): 
    '''
    Grafica frontera de decisión.
    '''
    margin = 0.5
    h = 0.01 
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    # Se genera una malla de puntos con la distancia h entre ellos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 

    # Se predice para todos los puntos de la malla
    Z = model.predict_boundaries(np.c_[xx.ravel(), yy.ravel()], boundary) 
    Z = Z.reshape(xx.shape) 
    
    # Se grafican los datos y la frontera de decisión
    ax.contourf(xx, yy, Z, cmap=plt.cm.PuOr) 
    #ax.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral) 

def plot_data_points_ax(ax, X_train, y_train, X_test, y_test):
    train_data = pd.DataFrame(np.column_stack((X_train, y_train)), columns=['x1','x2','y'])
    test_data = pd.DataFrame(np.column_stack((X_test, y_test)), columns=['x1','x2','y'])

    ax.plot(train_data['x1'][y_train == 1], 
            train_data['x2'][y_train == 1], 
            'o', alpha= 0.7, color = 'violet', label = 'Entrenamiento: Clase 1')
    ax.plot(test_data['x1'][y_test == 1], 
            test_data['x2'][y_test == 1], 
            'x', color = 'purple', label = 'Prueba: Clase 1')
    ax.plot(train_data['x1'][y_train == 0], 
             train_data['x2'][y_train == 0],
            'o', alpha= 0.7, color = 'sandybrown', label = 'Entrenamiento: Clase -1')
    ax.plot(test_data['x1'][y_test == 0], 
            test_data['x2'][y_test == 0], 
            'x', color = 'saddlebrown', label = 'Prueba: Clase -1')

def plot_data_points_ax2(ax, X_train, y_train, X_test, y_test):
    train_data = pd.DataFrame(np.column_stack((X_train, y_train)), columns=['x1','x2','y'])
    test_data = pd.DataFrame(np.column_stack((X_test, y_test)), columns=['x1','x2','y'])

    ax.plot(train_data['x1'][y_train == 0], 
             train_data['x2'][y_train == 0],
            'o', alpha= 0.7, color = 'sandybrown', label = 'Entrenamiento: Clase 0')
    ax.plot(test_data['x1'][y_test == 0], 
            test_data['x2'][y_test == 0], 
            'x', color = 'saddlebrown', label = 'Prueba: Clase 0')
    ax.plot(train_data['x1'][y_train == 1], 
            train_data['x2'][y_train == 1], 
            'o', alpha= 0.7, color = 'violet', label = 'Entrenamiento: Clase 1')
    ax.plot(test_data['x1'][y_test == 1], 
            test_data['x2'][y_test == 1], 
            'x', color = 'purple', label = 'Prueba: Clase 1')
    ax.plot(train_data['x1'][y_train == 2], 
            train_data['x2'][y_train == 2], 
            'o', alpha= 0.7, color = 'cornflowerblue', label = 'Entrenamiento: Clase 2')
    ax.plot(test_data['x1'][y_test == 2], 
            test_data['x2'][y_test == 2], 
            'x', color = 'darkblue', label = 'Prueba: Clase 2')

def plot_figure(model, X, X_train, y_train, X_test, y_test, epochs):
    n_samples  = X_train.shape[0]
    boundaries = model.boundaries
    # for j in range(n_samples): 
    #     anim_fig(X, X_train, y_train, X_test, y_test, hyperplanes[j], j, n_samples, out_folder)
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    fig.suptitle('Fronteras de decisión en diferentes épocas')
    obs = [1, int(epochs/2), epochs]

    for idx, ax in enumerate(axs.flat): 
        ind = obs[idx]-1
        plot_data_points_ax2(ax, X_train, y_train, X_test, y_test)
        ax.legend(loc = "lower left", fontsize=7)
        plot_decision_boundary(boundaries[idx], X, model, ax)
        ax.set_title(f'Epoch: {str(ind+1)}/{epochs}', fontsize=10)
    for ax in axs.flat:
        ax.set(xlabel='Característica 1', ylabel='Característica 2')
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

def plot_scores(scores): 
    accuracy = [] 
    loss = [] 
    for (x,y) in scores: 
        accuracy.append(x)
        loss.append(y)
    
    plt.figure(figsize=(6, 6))
    plt.title('Precisión y pérdida en diferentes épocas')
    plt.plot(loss, label='Loss', linewidth=2)
    plt.plot(accuracy, label='Accuracy', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.show()

def vectorized_result(j, neurons_output_layer):
    '''
    El número de clases a predecir debe corresponder con las neuronas en la capa de salida
    Parámetros: 
        neurons_output_layer: número de neuronas en la capa de salida
    Salida: 
        e: vector (1xn) con n el número de clases posibles. Tiene un 1 en la j-ésima posición siendo j la 
        clase a la que corresponde
    '''
    e = np.zeros((neurons_output_layer, 1))
    e[j] = 1.0
    return e