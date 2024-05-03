import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Función para calcular la distancia euclidiana entre dos vectores
def euclidean_distance(x, w):
    return np.linalg.norm(x - w)

# Función para encontrar el índice del nodo ganador (unidad de salida más cercana)
def find_winner(x, som_map):
    distances = [euclidean_distance(x, w) for w in som_map]
    return np.argmin(distances)

# Función para actualizar los pesos de las unidades de salida según el modelo de Gas Neural
def update_weights(x, som_map, winner, learning_rate, sigma):
    for i in range(len(som_map)):
        influence = neighborhood_function(np.abs(i - winner), sigma)
        som_map[i] += learning_rate * influence * (x - som_map[i])

# Función de función de vecindad (neighborhood function) para el ajuste de pesos
def neighborhood_function(distance, sigma):
    return np.exp(-distance**2 / (2 * sigma**2))

# Función principal para entrenar el modelo de Gas Neural
def train_gas_neural(data, som_map, signals, learning_rate, sigma, ax):
    scat = ax.scatter(data[:, 0], data[:, 1], c='b', s=2)
    som_scatter = ax.scatter(som_map[:, 0], som_map[:, 1], c='r')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    def update(frame):
        x = data[frame]
        winner = find_winner(x, som_map)
        update_weights(x, som_map, winner, learning_rate, sigma)
        som_scatter.set_offsets(som_map)
        print(f'Gas Neural - Signal {frame + 1}')
        return som_scatter,
    
    anim = FuncAnimation(ax.figure, update, frames=signals, interval=0.1, blit=True)
    plt.show()

# Función para generar datos aleatorios en un anillo
def generate_data(num_samples):
    radius = 0.5
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    x1 = (radius + 0.2 * np.random.rand(num_samples)) * np.cos(theta) + 0.5
    x2 = (radius + 0.2 * np.random.rand(num_samples)) * np.sin(theta) + 0.5
    data = np.column_stack((x1, x2))
    return data

# Parámetros del modelo
input_dim = 2
output_dim = 350
signals = 40000
learning_rate = 0.1
sigma = 0.05

# Inicializar el mapa de Gas Neural (unidades de salida)
som_map = np.random.rand(output_dim, input_dim)

# Generar datos aleatorios en un anillo
data = generate_data(signals)

# Crear la figura y el eje
fig, ax = plt.subplots()
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# Entrenar el modelo de Gas Neural con animación
train_gas_neural(data, som_map, signals, learning_rate, sigma, ax)
