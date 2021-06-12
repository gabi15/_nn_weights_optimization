import numpy
import ga
import pickle
from pso import pso, plot_pso
import statistics


f = open("dataset_features.pkl", "rb")
data_inputs2 = pickle.load(f)
f.close()
features_STDs = numpy.std(data_inputs2, axis=0)
data_inputs = data_inputs2[:, features_STDs>50]

f = open("outputs.pkl", "rb")
data_outputs = pickle.load(f)
f.close()

# parameters to play with, the bigger population the better results should be, but it takes memory,
# iterations should probably be about 10
# n_pop = 2
# iters = 2
# c1 = 2.3
# c2 = 2.3

# test_parameters = [{"n_pop": 50, "iters": 10, "c1": 2.0, "c2": 2.0}, {"n_pop": 40, "iters": 10, "c1": 2.0, "c2": 2.0},
#                    {"n_pop": 30, "iters": 10, "c1": 2.0, "c2": 2.0}, {"n_pop": 50, "iters": 10, "c1": 1.49445, "c2": 1.49445},
#                    {"n_pop": 40, "iters": 10, "c1": 1.49445, "c2": 1.49445}, {"n_pop": 30, "iters": 10, "c1": 1.49445, "c2": 1.49445}]

test_parameters = [{"n_pop": 40, "iters": 18, "c1": 2.0, "c2": 2.0}, {"n_pop": 40, "iters": 18, "c1": 1.49445, "c2": 1.49445},
                   {"n_pop": 40, "iters": 18, "c1": 2.8, "c2": 1.3}]
#test_parameters = [{"n_pop": 3, "iters": 2, "c1": 2.0, "c2": 2.0}, {"n_pop": 3, "iters": 2, "c1": 1.9, "c2": 1.46}]
for el in test_parameters:
    n_pop = el["n_pop"]
    iters = el["iters"]
    c1 = el["c1"]
    c2 = el["c2"]

    best_scores = []
    best_accuracies = []
    #loop to check various parameters
    for i in range(10):
        print(i)
        #Creating the initial population.
        initial_pop_weights = []

        for curr_sol in numpy.arange(0, n_pop):
            HL1_neurons = 150
            input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(data_inputs.shape[1], HL1_neurons))

            HL2_neurons = 60
            HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))

            output_neurons = 4
            HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))
            initial_pop_weights.append(numpy.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights], dtype=numpy.ndarray))

        pop_weights_mat = numpy.array(initial_pop_weights,  dtype=numpy.ndarray)
        pop_weights_vector = ga.mat_to_vector(pop_weights_mat)

        result = pso(c1, c2, iters, n_pop, pop_weights_vector, pop_weights_mat, data_inputs, data_outputs)
        #plot_pso(result[2], iters, n_pop, c1, c2)
        print("Accuracy of the best solution is : ", result[1])
        best_scores.append(result[1])
        best_accuracies.append(result[2])
    print(100*'-')
    print("Best score: {}".format(max(best_scores)))
    print("Mean: {}".format(statistics.mean(best_scores)))
    print("Stdev: {}".format(statistics.stdev(best_scores)))

    f = open("outputs/iters{}_population{}_c1_{}_c2_{}.txt".format(iters, n_pop, c1, c2), "w")
    f.write("Results for {} iters, population {}, c1 = {}, c2 = {}\n".format(iters, n_pop, c1, c2))
    f.write("Best score: {}\n".format(max(best_scores)))
    f.write("Worst score: {}\n".format(min(best_scores)))
    f.write("Mean: {}\n".format(statistics.mean(best_scores)))
    f.write("Stdev: {}\n".format(statistics.stdev(best_scores)))
    f.write("best scores:\n")
    f.write(str(best_scores))
    f.write("\naccuracies history:\n")
    f.write(str(best_accuracies))
    f.close()

    plot_pso(best_accuracies[best_scores.index(max(best_scores))], iters, n_pop, c1, c2)

