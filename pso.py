from numpy.random import uniform
from numpy.random import rand
from math import pi,sqrt,exp, inf
import copy
import statistics
import ANN
import ga
import numpy
import matplotlib.pyplot as plt


def pso(c1, c2, iters, n_pop, pop, pop_weights_mat, data_inputs, data_outputs):
    """
    optimizes weights for nn
    :param c1:
    :param c2:
    :param iters: number of iterations
    :param n_pop: how big populations is
    :param pop: initial population
    :param pop_weights_mat: population in form of matrices needed for nn
    :param data_inputs: file withs input for nn
    :param data_outputs: file with expected outputs of nn
    :return: 0: weights of the best solution, 1: accuracy of the best solution, 2: accuracies through iterations used
     for plotting
    """
    parameters_num = pop[0].size
    x_best = copy.copy(pop)
    v_pop = rand(n_pop, parameters_num)
    accuracies = []
    g_best = pop[0]
    g_best_val = -inf

    for count in range(iters):
        print(count)
        pop_weights_mat = ga.vector_to_mat(pop, pop_weights_mat)
        scores = ANN.fitness(pop_weights_mat, data_inputs, data_outputs, activation="sigmoid")
        print("fitness")
        print(scores)
        if g_best_val < max(scores):
            g_best = pop[numpy.argmax(scores)]
            g_best_val = max(scores)
        accuracies.append(g_best_val)
        print("g_best:")
        print(g_best_val)
        phi = c1+c2
        K = 2 / abs(2 - phi - sqrt(phi * phi - 4 * phi))
        # omega = 1
        omega_max = 0.9
        omega_min = 0.4
        omega = omega_max - ((omega_max - omega_min) / iters) * count
        for i, el in enumerate(pop):
            for j, par in enumerate(el):
                #omega = 1 + 0.5* exp(-1*(pop[i][j]-x_best[i][j]))
                v_pop[i][j] = K*(omega*v_pop[i][j] + uniform(0, c1) * (x_best[i][j] - pop[i][j]) + uniform(0, c2)*(g_best[j] - pop[i][j]))
                pop[i][j] += v_pop[i][j]

        # change local best
        pop_weights_best = ga.vector_to_mat(x_best, pop_weights_mat)
        pop_weights_matt = ga.vector_to_mat(pop, pop_weights_mat)
        best_scores = ANN.fitness(pop_weights_best, data_inputs, data_outputs, activation="sigmoid")
        some_scores = ANN.fitness(pop_weights_matt, data_inputs, data_outputs, activation="sigmoid")
        for i, el in enumerate(pop_weights_matt):
            if some_scores[i]>best_scores[i]:
                x_best[i]=pop[i]

    scores = ANN.fitness(pop_weights_mat, data_inputs, data_outputs, activation="sigmoid")
    if g_best_val < max(scores):
        g_best = pop[numpy.argmax(scores)]
        g_best_val = max(scores)
    accuracies.append(g_best_val)

    return g_best, g_best_val, accuracies


def plot_pso(accuracies, num_generations, population, c1, c2):
    plt.plot(accuracies, linewidth=4, color="black")
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Fitness", fontsize=15)
    plt.xticks(numpy.arange(0, num_generations+1, 1), fontsize=12)
    plt.yticks(numpy.arange(0, 101, 5), fontsize=12)
    plt.text(num_generations-2, 10, "iterations: {}\npopulation: {}\nc1={} c2={}".format(num_generations, population, c1, c2),
             fontsize=10, bbox=dict(boxstyle="round", ec=(0.7, 0.8, 1.0), fc=(0.8, 0.8, 1.0),))
    save_string = "outputs/iters{}_population{}_c1_{}_c2_{}.png".format(num_generations, population, c1, c2)
    plt.savefig(save_string)
    plt.close()


