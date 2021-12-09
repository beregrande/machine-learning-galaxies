import matplotlib.pyplot as plt 
import numpy as np

x = np.linspace(-5, 5)

'Definimos las diferentes funciones de activación'
def softmax(z): 
    expo = np.exp(z)
    expo_sum = np.sum(np.exp(z))
    return expo/expo_sum

def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z): 
    return np.tanh(z)

def relu(z): 
    return np.maximum(z, 0)

def leakyrelu(z): 
    alpha = 0.05
    return np.maximum(z, 0) + alpha*np.minimum(z, 0)

def absrelu(z): 
    alpha = -1
    return np.maximum(z, 0) + alpha*np.minimum(z, 0)

plt.plot(x, sigmoide(x), 'r', label='Sigmoide')
plt.plot(x, softmax(x), 'b', label='Softmax')
plt.title('Función sigmoide y softmax')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.legend()
plt.show()

plt.plot(x, absrelu(x), 'g', label = 'Absolute Value')
plt.plot(x, leakyrelu(x), 'b', label='Leaky ReLU')
plt.plot(x, relu(x), 'r', label='ReLU')
plt.title('Diferentes funciones ReLU')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.legend()
plt.show()

plt.plot(x, sigmoide(x), 'r', label='Sigmoide')
plt.plot(x, tanh(x), 'g', label='Tanh')
plt.title('Función sigmoide y tanh')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.legend()
plt.show()
