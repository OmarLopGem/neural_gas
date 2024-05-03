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

# Función para actualizar los pesos de las unidades de salida
def update_weights(x, som_map, winner, learning_rate):
    for i in range(len(som_map)):
        som_map[i] += learning_rate * neighborhood_function(i, winner) * (x - som_map[i])

# Función de función de vecindad (neighborhood function) para el ajuste de pesos
def neighborhood_function(idx, winner):
    sigma = 2.5  # Parámetro de ancho de vecindad
    distance = np.abs(idx - winner)
    return np.exp(-distance**2 / (2 * sigma**2))

# Función principal para entrenar el modelo de Gas Neural
def train_gas_neural(data, som_map, epochs, learning_rate, ax):
    scat = ax.scatter([], [], c='b', s=2)
    som_scatter = ax.scatter(som_map[:, 0], som_map[:, 1], c='r')
    circle_outer = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='-', linewidth=1)
    circle_inner = plt.Circle((0, 0), 0.7, color='gray', fill=False, linestyle='-', linewidth=1)
    ax.add_artist(circle_outer)
    ax.add_artist(circle_inner)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    def init():
        return som_scatter,

    def update(frame):
        x = data[frame]
        winner = find_winner(x, som_map)
        update_weights(x, som_map, winner, learning_rate)
        som_scatter.set_offsets(som_map)
        ax.set_title(f'Gas Neural - Epoch {frame + 1}')
        return som_scatter,

    anim = FuncAnimation(ax.figure, update, frames=epochs, init_func=init, interval=1, blit=True)
    plt.show()

# Función para generar datos aleatorios en un anillo
def generate_data(num_samples):
    radius = np.sqrt(np.random.rand(num_samples))  # Radio aleatorio en el rango [0, 1)
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    x1 = radius * np.cos(theta)
    x2 = radius * np.sin(theta)
    data = np.column_stack((x1, x2))
    return data

# Parámetros del modelo
input_dim = 2
output_dim = 400
epochs = 40000
learning_rate = 0.1

# Inicializar el mapa de Gas Neural (unidades de salida)
som_map = np.random.rand(output_dim, input_dim) * 2 - 1


# Generar datos aleatorios en un anillo
data = generate_data(epochs)

# Crear la figura y el eje
fig, ax = plt.subplots()
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# Entrenar el modelo de Gas Neural con animación
train_gas_neural(data, som_map, epochs, learning_rate, ax)
