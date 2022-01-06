import numpy as np
import random
from numpy.core.fromnumeric import shape, size

class Network: 
    def __init__(self, neurons, learning_rate): 
        self.learning_rate = learning_rate
        self.num_layers = len(neurons)
        self.neurons = neurons    #Lista con número de neuronas por capa
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(neurons[:-1], neurons[1:])]
        self.biases = [np.random.randn(y, 1) for y in neurons[1:]]
        self.linear_outputs = []   # Lista de lo hiperplanos z por capa 
        self.activations = []      # Lista de todas las activaciones por capa
        self.activ_func_hid = self.tanh     #Función de activación capas ocultas
        self.activ_func_out = self.sigmoid    #Función de activación capa de salida
        self.dfunc_hid = self.dTanh     #Derivada función activación capas ocultas 
        self.dfunc_out = self.dSigmoid        #Derivada función activación capas ocultas
        self.scores = []
        self.boundaries = []

    def feedforward(self, x):
        '''
        Propaga hacia adelante la información de la observación de entrada. 
        Guarda las salidas lineales y las activaciones de cada capa. 
        Parámetros: 
            x: vector de entrada de la red
        Salida: 
            y_hay: predicción de la red para la entrada x
        '''
        a = x
        self.activations.append(a)
        for b, w in zip(self.biases, self.weights):
            linear_output = np.dot(w, a)+b
            
            last_layer = (b[0][0] == self.biases[-1][0][0]) 
            if last_layer==False: 
                a = self.activ_func_hid(linear_output) #Se calcula la salida de cada capa de neuronas aplicando la función
            else:
                a = self.activ_func_out(linear_output)   
                y_hat = a
            
            self.linear_outputs.append(linear_output)
            self.activations.append(a)
        return y_hat

    def train(self, train_data, epochs, eta, test_data=None, SGD=False, mini_batch_size=None):
        '''
        Entrena la red neuronal utilizando descenso por gradiente. Imprime precisión y pérdida final. 
        Parámetros: 
            train_data: lista de tuplas (x, y) de datos de entrenamiento 
            epochs: número de veces que se visitan los datos de entrenamiento
            eta: tasa de aprendizaje para descenso por gradiente
            SGD : booleano que indica si se implementa descenso por gradiente estocástico
            mini_batch_size: número de observaciones por cada mini_batch si SGD=True
        '''
        if test_data: n_test = len(test_data)
        n = len(train_data)
        e = epochs / 5
        for j in range(epochs):
            #Descenso por gradiente estocástico
            if SGD==True: 
                random.shuffle(train_data)
                mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                for mini_batch in mini_batches:
                    self.gradient_descent(mini_batch, eta) #Actuzaliza los pesos de los mini_batches
                if j == 1 or j == int(epochs/2) or j+1 ==epochs: 
                    self.boundaries.append(zip(self.biases, self.weights))
            else: 
                #Descenso por gradiente
                self.gradient_descent(train_data, eta) #Actualiza los pesos con todo el conjunto de datos
                if j == 1 or j == int(epochs/2) or j ==epochs: 
                    self.boundaries.append(zip(self.biases, self.weights))

            if test_data:
                accuracy = self.accuracy(test_data)
                loss = self.loss(test_data)
                self.scores.append((accuracy, loss))
                if j % e == 0: 
                    print("Epoch : %2d --> Accuracy: %f Loss: %f" % 
                    (j, accuracy, loss))
                elif j+1 == epochs: 
                    print("Epoch complete --> Accuracy: %f Loss %f" %
                    (accuracy, loss))
            else:
                print("Epoch %2d complete" % (j))

    def gradient_descent(self, batch, eta):
        '''
        Actualiza los pesos y los sesgos aplicando descenso por gradiente, calculando el gradiente con
        backporpagation para cada observación del batch de información. 
        Parámetros: 
            batch: lista de tuplas (x, y). Puede ser todo el conjunto de entrenamiento o un mini_batch del conjunto (SGD)
            eta: tasa de aprendizaje
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  #Se actualizan los valores de nabla 
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w-(eta/len(batch))*nw                          #Se actualizan los pesos y baises
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(batch))*nb 
                        for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        '''
        Calcula el gradiente utilizando la regla de la cadena
        Parámetros: 
            x: observación del conjunto de datos
            y: etiqueta de la observación
        Salida: 
            (nabla_b, nabla_w): tupla que representa el gradiente para la función de costo. Cada elemento de la tupla
            es una lista de matrices, cada matriz es el gradiente calculado para cada capa. (capa por capa)
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #Feedforward
        self.feedforward(x) 
        # Se calcula el error proporcional a cada peso con la regla de la cadena 
        # Primero se calcula para la última capa 
        delta = self.dLoss(self.activations[-1], y)*self.dfunc_out(self.linear_outputs[-1]) #Regla de la cadena 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, self.activations[-2].transpose()) 
        
        #Se propaga a las otras capas
        for l in range(2, self.num_layers):
            linear_output = self.linear_outputs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta)*self.dfunc_hid(linear_output) #Regla de la cadena
            nabla_b[-l] = delta 
            nabla_w[-l] = np.dot(delta, self.activations[-l-1].transpose())  
        return (nabla_b, nabla_w)
    
    def accuracy(self, test_data):
        '''
        Caclula el accuracy del modelo, sumando el número de observaciones del conjunto de entrenamiento
        en las que la red obtiene el resultado correcto y dividiendo entre la cantidad total de obs. 
        Parámetros: 
            test_data: lista de tuplas (x, y) con los datos de prueba
        Salida: 
            accuracy: score que evalúa el modelo
        '''
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        n_test = len(test_results)
        accuracy = sum(int(x == y) for (x, y) in test_results) / n_test
        return accuracy

    def predict(self, X): 
        '''
        Parámetros:
            X: matriz con observaciones a predecir
        Salida: 
            y_predicted: vector de predicciones de la matriz
        '''
        a = self.feedforward(X.T)
        y_predicted = a.T.argmax(axis=1)
        return y_predicted
    
    def predict_boundaries(self, X, boundary): 
        a = X.T
        for b, w in boundary:
            linear_output = np.dot(w, a)+b
            
            last_layer = (b[0][0] == self.biases[-1][0][0]) 
            if last_layer==False: 
                a = self.activ_func_hid(linear_output) #Se calcula la salida de cada capa de neuronas aplicando la función
            else:
                a = self.activ_func_out(linear_output)   
                #y_hat = a

        y_predicted = a.T.argmax(axis=1)
        return y_predicted

    "Funciones auxiliares"
    def loss(self, test_data): 
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        costo = [(x - y)**2 for (x, y) in test_results]          
        loss = np.mean(costo)
        return loss
    
    def dLoss(self, y_pred, y_real):
        return (y_pred - y_real)
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def dSigmoid(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def relu(self, z):
        return np.maximum(0, z)

    def dRelu(self, z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z

    def tanh(self, z): 
        return np.tanh(z)
    
    def dTanh(self, z): 
        return 1 - self.tanh(z)**2
        
    def softmax(self, z): 
        return np.exp(z)/np.sum(np.exp(z))

    def dSoftmax(self,z): 
        s = self.softmax(z)
        # s_vector = s.reshape(s.shape[0],1)
        # s_matrix = np.tile(s_vector,s.shape[0])
        # return np.diag(s) - (s_matrix * np.transpose(s_matrix))     
        pass        