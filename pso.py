from numpy.random import uniform
from numpy.random import rand
from math import cos
from math import pi,sqrt,exp, inf
import copy
import statistics
import ANN
import ga
import numpy


def fitness(x_pop):
    x = x_pop[0]
    y = x_pop[1]
    return x*x + y*y - 20*(cos(pi*x)+cos(pi*y)-2)


def pso(c1, c2, iters, n_pop, pop, pop_weights_mat, data_inputs, data_outputs):


    #x_pop = uniform(-10, 10, (n_pop, 2)
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
        # phi = c1+c2
        # omega = 2 / (2 + sqrt(phi * phi - 4 * phi))
        # omega = 1
        omega_max = 0.9
        omega_min = 0.4
        omega = 0.9 - ((omega_max - omega_min) / iters) * count
        for i, el in enumerate(pop):
            for j, par in enumerate(el):
                #w = 1 + 0.5* exp(-1*(pop[i][j]-x_best[i][j]))
                v_pop[i][j] = omega*v_pop[i][j] + uniform(0, c1) * (x_best[i][j] - pop[i][j]) + uniform(0, c2)*(g_best[j] - pop[i][j])
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
    best_score = max(scores)
    print(best_score)
    print(pop)
    #best_coordinate = pop[scores.index(best_score)]
    return best_score, accuracies


if __name__ == "__main__":
    n_pop = 1000
    iters = 6
    best_scores = []
    for i in range(10):
        print(100*"-")
        result = pso(2.5, 2.5, iters, n_pop)
        best_scores.append(result[1])
        print("Best coordinates: {}\nBest score: {}".format(result[0], result[1]))

    print(100*'-')
    print("Best score: {}".format(min(best_scores)))
    print("Mean: {}".format(statistics.mean(best_scores)))
    print("Stdev: {}".format(statistics.stdev(best_scores)))
