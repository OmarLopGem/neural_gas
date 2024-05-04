import random
import math
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt

def plot_weights(weights, e, iteration):
    plt.close()
    outer_circle = plt.Circle((0.5, 0.5), 0.5, color='gray', fill=False)
    inner_circle = plt.Circle((0.5, 0.5), 0.3, color='gray', fill=False)
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjusted size
    # Add title
    ax.set_title(f"t_max at {iteration+1}" if iteration != 0 else "t_max at 0")
    ax.add_patch(outer_circle)
    ax.add_patch(inner_circle)
    
    # Generate unique colors for each neuron
    colors = [plt.cm.jet(i/float(len(weights))) for i in range(len(weights))]
    
    for i, (x, y) in enumerate(weights):
        plt.scatter(x, y, color=colors[i])
    
    plt.draw()
    plt.show()

def neural_gas(num_signals, num_neurons, t_max):
    
    lambda_i = 10
    lambda_f = 0.01
    epsilon_i = 0.5
    epsilon_f = 0.05

    # Initialization of weights
    weights = []
    i = 0
    while len(weights) < num_neurons:
        w = (random.random(), random.random())
        if (w[0]-(0.5))**2+(w[1]-(0.5))**2 > 0.5**2 or (w[0]-(0.5))**2+(w[1]-(0.5))**2 < 0.3**2:
            continue
        else:
            weights.append(w)

    # Main loop
    i = 0
    while i < t_max:
        # Generate random signal
        e = (random.random(), random.random())
        if (e[0]-(0.5))**2+(e[1]-(0.5))**2 > 0.5**2 or (e[0]-(0.5))**2+(e[1]-(0.5))**2 < 0.3**2:
            continue
        else:
            distances = []
            for j in range(len(weights)):
                distance = math.sqrt((weights[j][0] - e[0])**2 + (weights[j][1] - e[1])**2)
                distances.append((distance, j))
            distances.sort()
            for k in range(len(distances)):
                # Update weights
                epsilon_t = (epsilon_i*(epsilon_f/epsilon_i)**((i+1)/t_max))
                lambda_t = (lambda_i*(lambda_f/lambda_i)**((i+1)/t_max))
                
                # Calculate the weight update delta based on the learning rate (epsilon_t), distance (lambda_t), and difference between the signal and weight position
                w_delta = epsilon_t * math.exp((-k) / lambda_t)
                w_delta_x = w_delta * (e[0] - weights[distances[k][1]][0])
                w_delta_y = w_delta * (e[1] - weights[distances[k][1]][1])
                
                # Update the weight at index distances[k][1] by adding the weight update delta to its x and y coordinates
                weights[distances[k][1]] = (weights[distances[k][1]][0]+w_delta_x, weights[distances[k][1]][1]+w_delta_y)
            if i == 0 or i == 299 or i == 2499 or i == 39999:
                plot_weights(weights, e, i)
        i += 1

if __name__ == "__main__":

    neural_gas(200, 400, 40000)
