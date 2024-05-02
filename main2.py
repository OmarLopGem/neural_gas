import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
import math

def pesos(n):
    return [(random.random(), random.random()) for _ in range(n)]

def senalesPesos(n, t_max):
    learning_rate = 0.1
    contador = 0
    posiciones = []
    pe = pesos(n)
    #print(pe)
    distanciasOrdenadas = []
    pos = 0
    for _ in range(t_max):

        e = (random.random(), random.random())

        for punto in pe:
            
            dist = math.sqrt((punto[0] - e[0])**2 + (punto[1] - e[1])**2)
            tupla = (dist, pos)
            pos += 1
            distanciasOrdenadas.append(tupla)

        distanciasOrdenadas.sort()

        for i, tup in enumerate(distanciasOrdenadas):
            distanciasOrdenadas[tup[1]] += learning_rate * (e - distanciasOrdenadas[tup[1]]) * np.exp(-i)


    print(distanciasOrdenadas)

        

    


def dibujarCirculo(r1, r2):
    # Definir el radio del círculo
    x = np.linspace(0, 1, 100)  

    # Ecuación del primer círculo centrado en (0.5, 0.5) con radio r1
    y1 = np.sqrt(r1**2 - (x - 0.5)**2) + 0.5 

    # Ecuación del segundo círculo centrado en el mismo punto (0.5, 0.5) con radio r
    radio1 = 0.5
    radio2 = 0.25
    centro_x = 0.5
    centro_y = 0.5

    theta = np.linspace(0, 2*np.pi, 100)
    x1 = centro_x + radio1 * np.cos(theta)
    y1 = centro_y + radio1 * np.sin(theta)

    theta = np.linspace(0, 2*np.pi, 100)
    x2 = centro_x + radio2 * np.cos(theta)
    y2 = centro_y + radio2 * np.sin(theta)

    # Dibujar los círculos
    plt.figure()
    plt.plot(x1, y1, 'b')
    plt.plot(x2, y2, 'b')
    plt.gca().set_aspect('equal', adjustable='box')  # Aspecto igual en ambos ejes para que los círculos se vean circulares
    plt.xlim(0, 1)  # Establecer límites del eje x
    plt.ylim(0, 1)  # Establecer límites del eje y
    plt.title('Dos círculos centrados en el mismo punto')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.grid(True)
    plt.show()

# Parametros
n = 3
dimension = 2
t_max = 5
r1 = 0.5
r2 = 0.25


dibujarCirculo(r1, r2)
senalesPesos(n, t_max)
