import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_weights(pesos, e, iteracion):
    plt.close()
    círculo_exterior = plt.Circle((0.5, 0.5), 0.5, color='gray', fill=False)
    círculo_interior = plt.Circle((0.5, 0.5), 0.3, color='gray', fill=False)
    fig, ax = plt.subplots(figsize=(6, 6))  # Tamaño ajustado
    #Add title
    ax.set_title(f"t_max en {iteracion+1}" if iteracion != 0 else "t_max en 0")
    ax.add_patch(círculo_exterior)
    ax.add_patch(círculo_interior)
    
    # Generar colores únicos para cada neurona
    colores = [plt.cm.jet(i/float(len(pesos))) for i in range(len(pesos))]
    
    for i, (x, y) in enumerate(pesos):
        plt.scatter(x, y, color=colores[i])
    
    plt.draw()
    plt.show()

def gas_neural(num_señales, num_neuronas, t_max):
    
    lambda_i = 10
    lambda_f = 0.01
    epsilon_i = 0.5
    epsilon_f = 0.05

    # Inicialización de pesos
    pesos = []
    i = 0
    while len(pesos) < num_neuronas:
        w = (random.random(), random.random())
        if (w[0]-(0.5))**2+(w[1]-(0.5))**2 > 0.5**2 or (w[0]-(0.5))**2+(w[1]-(0.5))**2 < 0.3**2:
            continue
        else:
            pesos.append(w)

    # Bucle principal
    i = 0
    while i < t_max:
        # Generación de señal aleatoria
        e = (random.random(), random.random())
        if (e[0]-(0.5))**2+(e[1]-(0.5))**2 > 0.5**2 or (e[0]-(0.5))**2+(e[1]-(0.5))**2 < 0.3**2:
            continue
        else:
            distancias = []
            for j in range(len(pesos)):
                distancia = math.sqrt((pesos[j][0] - e[0])**2 + (pesos[j][1] - e[1])**2)
                distancias.append((distancia, j))
            distancias.sort()
            for k in range(len(distancias)):
                # Actualización de pesos
                epsilon_para_t = (epsilon_i*(epsilon_f/epsilon_i)**((i+1)/t_max))
                lambda_para_t = (lambda_i*(lambda_f/lambda_i)**((i+1)/t_max))
                w_delta = epsilon_para_t*(math.exp((-k)/lambda_para_t))
                w_delta_x = w_delta*(e[0]-pesos[distancias[k][1]][0])
                w_delta_y = w_delta*(e[1]-pesos[distancias[k][1]][1])
                pesos[distancias[k][1]] = (pesos[distancias[k][1]][0]+w_delta_x, pesos[distancias[k][1]][1]+w_delta_y)
            
            if i == 0 or i == 299 or i == 2499 or i == 39999:
                plot_weights(pesos, e, i)
        i += 1

# Ejemplo de uso con 200 señales y 50 neuronas
gas_neural(200, 400, 40000)
