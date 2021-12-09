import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split

def data_loader(data_type): 
    '''
    Carga los tipos de datos y divide los conjuntos de entrenamiento y prueba'''
    if data_type == 0: 
        X, y = make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.2, random_state=697)
        y = np.array([1 if i > 0 else -1 for i in y])
    elif data_type == 1: 
        X, y = make_circles(200, noise = .1, random_state =697, factor = .4)
        y = np.array([1 if i > 0 else -1 for i in y])
    elif data_type == 2: 
        X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=1.2, random_state=501)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return (X, y, X_train, y_train, X_test, y_test)

'''Funciones para graficar los datos y las fronteras de decisión'''
def plot_data_points_bin(ax, X_train, y_train, X_test, y_test):
    '''Grafica el conjunto de datos dividiéndolo en conjunto de entrenamiento y 
    conjunto de prueba.'''
    train_data = pd.DataFrame(np.column_stack((X_train, y_train)), columns=['x1','x2','y'])
    test_data = pd.DataFrame(np.column_stack((X_test, y_test)), columns=['x1','x2','y'])

    ax.plot(train_data['x1'][y_train == 1], 
            train_data['x2'][y_train == 1], 
            'o', alpha= 0.7, color = 'violet', label = 'Entrenamiento: Clase 1')
    ax.plot(test_data['x1'][y_test == 1], 
            test_data['x2'][y_test == 1], 
            'x', color = 'purple', label = 'Prueba: Clase 1')
    ax.plot(train_data['x1'][y_train == -1], 
             train_data['x2'][y_train == -1],
            'o', alpha= 0.7, color = 'sandybrown', label = 'Entrenamiento: Clase -1')
    ax.plot(test_data['x1'][y_test == -1], 
            test_data['x2'][y_test == -1], 
            'x', color = 'saddlebrown', label = 'Prueba: Clase -1')

def plot_data_points_mult(ax, X_train, y_train, X_test, y_test):
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

def plot_decision_boundary(hyperplane, X, ax=None): 
    margin = 0.5
    h = 0.01 
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    # Se genera una malla de puntos con la distancia h entre ellos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    if ax: 
        line, = ax.plot([], [], 'r', linewidth=0.8)
    else: 
        line, = plt.plot([], [], 'r', linewidth=0.8)

    m = hyperplane[1]
    b = hyperplane[0]
    y_ = b - m * xx
    line.set_data(xx , y_)

'''Funciones auxiliares para crear animación'''
def plot_data_points(X_train, y_train, X_test, y_test):
    train_data = pd.DataFrame(np.column_stack((X_train, y_train)), columns=['x1','x2','y'])
    test_data = pd.DataFrame(np.column_stack((X_test, y_test)), columns=['x1','x2','y'])

    plt.plot(train_data['x1'][y_train == 1], 
            train_data['x2'][y_train == 1], 
            'o', color = 'violet', label = 'Entrenamiento: Clase 1')
    plt.plot(test_data['x1'][y_test == 1], 
            test_data['x2'][y_test == 1], 
            'x', color = 'purple', label = 'Prueba: Clase 1')
    plt.plot(train_data['x1'][y_train == -1], 
             train_data['x2'][y_train == -1],
            'o', color = 'sandybrown', label = 'Entrenamiento: Clase -1')
    plt.plot(test_data['x1'][y_test == -1], 
            test_data['x2'][y_test == -1], 
            'x', color = 'saddlebrown', label = 'Prueba: Clase -1')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')

def anim_fig(X, X_train, y_train, X_test, y_test, hyperplane, j, n_samples, out_folder):
    #ffmpeg
    mask = np.ones((n_samples,), dtype=bool)
    mask[j] = False
    plot_data_points(X_train[mask, :], y_train[mask], X_test, y_test)
    plt.scatter(X_train[j, 0], X_train[j, 1], s=50, c='lime', marker="D", label="Observación")
    plt.legend(loc = "upper right", fontsize=7)
    plot_decision_boundary(hyperplane, X)
    plt.title(f' Observación: {j+1}/{n_samples}', fontsize = 10)

    plt.savefig(f'{out_folder}/frame{j}.png')
    plt.close()

'''Función para gráficar las fronteras de decisión en diferentes observaciones'''
def plot_figure(model, data_type, X, X_train, y_train, X_test, y_test, anim=False, out_folder=None):
    if data_type == 2: 
        plot_data_points_ax = plot_data_points_mult
    else: 
        plot_data_points_ax = plot_data_points_bin

    n_samples  = X_train.shape[0]
    hyperplanes = model.hyperplanes
    if anim: 
        for j in range(n_samples): 
            anim_fig(X, X_train, y_train, X_test, y_test, hyperplanes[j], j, n_samples, out_folder)

    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    fig.suptitle('Fronteras de decisión en diferentes observaciones')
    obs = [1, int(n_samples/2), n_samples]

    for idx, ax in enumerate(axs.flat): 
        mask = np.ones((n_samples,), dtype=bool)
        ind = obs[idx]-1
        mask[ind] = False
        plot_data_points_ax(ax, X_train[mask, :], y_train[mask], X_test, y_test)
        ax.scatter(X[ind, 0], X[ind, 1], s=50, c='lime', marker="D", alpha= 1, label="Observación")
        ax.legend(loc = "lower left", fontsize=7)
        plot_decision_boundary(hyperplanes[ind], X, ax)
        ax.set_title(f'Observación: {str(ind+1)}/{n_samples}', fontsize=10)
    for ax in axs.flat:
        ax.set(xlabel='Característica 1', ylabel='Característica 2')
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

def plot_scores(scores): 
    accuracy = [] 
    losses = []
    for s in scores: 
        acc = s[0]
        loss = s[1]
        accuracy.append(acc)
        losses.append(loss)
    plt.figure(figsize=(4, 4))
    plt.title('Precisión y pérdida')
    plt.plot(accuracy, label='Precisión', linewidth=2)
    plt.plot(losses, label='Pérdida', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.show()


