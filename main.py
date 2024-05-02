import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def initialize_neurons(n, input_dim, inner_radius=0.8, outer_radius=0.7):
    """
    Inicializa los vectores de referencia de los nodos para n unidades dentro de un anillo.
    """
    x_inner = np.random.uniform(-1, 1, n)
    y_inner = np.random.uniform(-1, 1, n)
    x_outer = np.random.uniform(-1, 1, n)
    y_outer = np.random.uniform(-1, 1, n)
    
    print(np.column_stack((x_inner, y_inner)), np.column_stack((x_outer, y_outer)))
    
    #Plot the values
    plt.plot(x_inner, y_inner, 'ro')
    plt.plot(x_outer, y_outer, 'bo')
    plt.show()
    input()
    
    return np.column_stack((x_inner, y_inner)), np.column_stack((x_outer, y_outer))

def find_closest_neurons(neurons, signal):
    """
    Encuentra los índices de los nodos más cercanos a la señal de entrada.
    """
    distances = np.linalg.norm(neurons - signal, axis=1)
    return np.argsort(distances)

def adjust_neurons(neurons, signal, closest_indices, learning_rate, inner_radius, outer_radius):
    """
    Ajusta los vectores de referencia de los nodos según la regla del gas neuronal,
    asegurándose de que permanezcan dentro del anillo definido por los radios internos y externos.
    """
    for i, idx in enumerate(closest_indices):
        neurons[idx] += learning_rate * (signal - neurons[idx]) * np.exp(-i)
        # Ajustar las posiciones de las neuronas para mantenerlas dentro del anillo
        distance_from_origin = np.linalg.norm(neurons[idx])
        if distance_from_origin < inner_radius:
            neurons[idx] *= inner_radius / distance_from_origin
        elif distance_from_origin > outer_radius:
            neurons[idx] *= outer_radius / distance_from_origin
    return neurons

# Parámetros
n_neurons = 10  # Número de neuronas
input_dim = 2  # Dimensión de la señal de entrada (para visualización en 2D)
t_max = 20  # Número máximo de iteraciones
learning_rate = 0.05  # Tasa de aprendizaje
inner_radius = 0.5  # Radio interno del anillo
outer_radius = 0.8  # Radio externo del anillo

# Inicialización de los nodos dentro de un anillo
neurons_inner, neurons_outer = initialize_neurons(n_neurons, input_dim, inner_radius, outer_radius)

# Crear la figura y los ejes
fig, ax = plt.subplots()
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

# Dibujar el anillo que delimita el área
inner_circle = plt.Circle((0, 0), inner_radius, color='blue', fill=False)
outer_circle = plt.Circle((0, 0), outer_radius, color='blue', fill=False)
ax.add_artist(inner_circle)
ax.add_artist(outer_circle)
        
# Inicializar la línea que representa las neuronas
neuron_lines_inner = [ax.plot([], [], marker='o')[0] for _ in range(n_neurons)]
neuron_lines_outer = [ax.plot([], [], marker='o')[0] for _ in range(n_neurons)]

# Función de inicialización de la animación
def init():
    for line_inner, line_outer in zip(neuron_lines_inner, neuron_lines_outer):
        line_inner.set_data([], [])
        line_outer.set_data([], [])
    return neuron_lines_inner + neuron_lines_outer

# Función de actualización de la animación
def update(frame):
    global neurons_inner, neurons_outer
    random_signal = np.random.rand(input_dim) * 2 - 1  # Señal dentro del rango [-1, 1]
    closest_indices_inner = find_closest_neurons(neurons_inner, random_signal)
    neurons_inner = adjust_neurons(neurons_inner, random_signal, closest_indices_inner, learning_rate, inner_radius, outer_radius)
    closest_indices_outer = find_closest_neurons(neurons_outer, random_signal)
    neurons_outer = adjust_neurons(neurons_outer, random_signal, closest_indices_outer, learning_rate, inner_radius, outer_radius)
    print("Iteración:", frame)  # Imprimir el número de iteración
    for i, (line_inner, line_outer) in enumerate(zip(neuron_lines_inner, neuron_lines_outer)):
        line_inner.set_data(neurons_inner[i, 0], neurons_inner[i, 1])
        line_outer.set_data(neurons_outer[i, 0], neurons_outer[i, 1])
        
    return neuron_lines_inner + neuron_lines_outer

# Crear la animación
ani = FuncAnimation(fig, update, frames=range(t_max), blit=False, interval=150, repeat=False)  # intervalo de 1 milisegundo

plt.title('Evolución de las neuronas (en un anillo)')
plt.xlabel('Dimensión X')
plt.ylabel('Dimensión Y')
plt.grid(True)
plt.show()
