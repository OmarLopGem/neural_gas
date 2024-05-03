import matplotlib.pyplot as plt
import numpy as np
import random
import math

def generate_random_points_between_circles(center, radius1, radius2, num_points):
    """
    Genera puntos aleatorios dentro del área entre dos círculos concéntricos.
    """
    points = []
    for _ in range(num_points):
        theta = random.uniform(0, 2*np.pi)
        r = random.uniform(radius2, radius1)  # Generar r dentro del área entre los dos círculos
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        points.append((x, y))
    return points

def update_weights_gas(weights, learning_rate, e, winners_idx):
    """
    Actualiza los pesos de acuerdo a los ganadores y al vector de entrada e usando la regla de gas neural.
    """
    for i in range(len(weights)):
        delta = np.array(e) - weights[i]
        weights[i] += learning_rate * np.exp(-np.linalg.norm(i - winners_idx)) * delta

def plot_circles(center, radius1, radius2):
    """
    Dibuja dos círculos.
    """
    theta = np.linspace(0, 2*np.pi, 100)
    x1 = center[0] + radius1 * np.cos(theta)
    y1 = center[1] + radius1 * np.sin(theta)

    x2 = center[0] + radius2 * np.cos(theta)
    y2 = center[1] + radius2 * np.sin(theta)

    plt.plot(x1, y1, 'b')
    plt.plot(x2, y2, 'b')

def plot_neurons(ax, weights):
    """
    Dibuja los pesos como neuronas.
    """
    x, y = zip(*weights)
    ax.scatter(x, y, color='red')

def gas_algorithm(center, radius1, radius2, num_neurons, t_max, learning_rate):
    """
    Implementación de la regla de gas neural para generar y actualizar los pesos.
    """
    fig, ax = plt.subplots()
    
    weights = generate_random_points_between_circles(center, radius1, radius2, num_neurons)
    plot_circles(center, radius1, radius2)
    plot_neurons(ax, weights)
    ax.set_aspect('equal', adjustable='box')  
    ax.set_xlim(0, 1)  
    ax.set_ylim(0, 1)  
    ax.set_title('Iteración 0')
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.grid(True)
    plt.show(block=False)
    
    for i in range(1, t_max+1):
        e = (random.uniform(0, 1), random.uniform(0, 1))  # Vector de entrada aleatorio
        
        # Encontrar los ganadores (neuronas más cercanas)
        distances = [np.linalg.norm(np.array(e) - np.array(w)) for w in weights]
        sorted_indices = np.argsort(distances)
        winners_idx = sorted_indices[:2]
        
        # Actualizar los pesos
        update_weights_gas(weights, learning_rate, e, winners_idx)
        
        # Actualizar la gráfica
        ax.clear()
        plot_circles(center, radius1, radius2)
        plot_neurons(ax, weights)
        ax.set_aspect('equal', adjustable='box')  
        ax.set_xlim(0, 1)  
        ax.set_ylim(0, 1)  
        ax.set_title('Iteración {}'.format(i))
        ax.set_xlabel('Eje X')
        ax.set_ylabel('Eje Y')
        ax.grid(True)
        plt.pause(0.1)  # Pausa para mostrar la actualización
    plt.show()

# Parámetros
center = (0.5, 0.5)
radius1 = 0.5
radius2 = 0.25
num_neurons = 400
t_max = 40000
learning_rate = 0.1

# Ejecutar el algoritmo de gas neural
gas_algorithm(center, radius1, radius2, num_neurons, t_max, learning_rate)

