import numpy as np 

class Perceptron: 
    def __init__(self, learning_rate=0.01): 
        self.learning_rate = learning_rate
        self.activation_func = self.step_func
        self.weights = None
        self.bias = None
        self.hyperplanes = []
        self.scores = []
        
    def train(self, X, y, epochs, test_data = None): 
        '''
        Entrenamiento del Perceptron. Actualiza los parámetros hasta que el algoritmo converja y guarda la información 
        de los hiperplanos que se generan. 
        Parámetros: 
            X: matriz con las observaciones del conjunto de datos.
            y: vector de etiquetas. 
        '''
        n, m = X.shape      #matriz de nxm (n observaciones y m características)
        self.weights = np.zeros(m)
        self.bias = 0 
        y_ = np.array([1 if i > 0 else -1 for i in y])  #se transforman las etiquetas a un vector de 0's y 1's"
        
        #El algoritmo converge cuando ya no hay error de clasifiación "
        e = epochs / 5
        for j in range(epochs):
            for idx, x_i in enumerate(X):    
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output) 

                if y_[idx]*y_predicted <= 0:     #tienen el signo contrario 
                    update = self.learning_rate * y_[idx]
                    self.weights += update * x_i  
                    self.bias += update
                
                if idx!= 0: 
                    self.hyperplanes.append((-self.bias / self.weights[1], self.weights[0] / self.weights[1])) 

            if test_data:
                accuracy = self.accuracy(test_data)
                loss = self.loss_func(self.hingeloss(test_data))
                self.scores.append([accuracy, loss])
                if j % e == 0: 
                    print("Epoch : %2d --> Accuracy: %f Loss %f" % 
                    (j+1, accuracy, loss))
                elif j+1 == epochs: 
                    print("Epoch complete --> Accuracy: %f Loss %f" %
                    (accuracy, loss))
            else:
                print("Epoch %2d complete" % (j))

    def predict(self, X): 
        '''
        Calcula las predicciones de la matriz X con los pesos obtenidos en el entrenamiento. 
        Parámetros: 
            X: matriz con observaciones a predecir.
        Salida: 
            y_predicted: vector con etiquetas predichas.
        '''
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
    
    def accuracy(self, test_data): 
        '''
        Calcula la métrica de accuracy para calificar el desempeño del modelo. 
        Parámetros: 
            y_true: vector con etiquetas reales. 
            y_pred: vector con etiquetas predichas.  
        '''
        X_test = test_data[0] 
        y_test = test_data[1]
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy
 
    def hingeloss(self, test_data): 
        X = test_data[0] 
        y = test_data[1]
        y_predicted = self.predict(X)
        z = y * y_predicted
        return np.where(z < 1, 1 - z, 0)

    def loss_func(self, hingeloss): 
        return np.mean(hingeloss)

    def step_func(self, X): 
        '''
        Función escalonada. Se utiliza como función de activación, se aplica a la salida lineal de la suma ponderada. 
        Parámetros:  
            X: valor de la salida lineal de una observación o arreglo de las salidas lineales de un conjunto de
            observaciones en la función predict.
        Salida: 
            1 si la entrada es > 0 y -1 e.o.c
        '''
        return np.where(X > 0, 1, -1)