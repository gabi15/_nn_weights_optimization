import numpy
import ga
import pickle
import ANN
import matplotlib.pyplot as plt
from pso import pso


def plot_pso(accuracies, num_generations):
    plt.plot(accuracies, linewidth=5, color="black")
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Fitness", fontsize=20)
    plt.xticks(numpy.arange(0, num_generations+1, 100), fontsize=15)
    plt.yticks(numpy.arange(0, 101, 5), fontsize=15)
    plt.show()


f = open("dataset_features.pkl", "rb")
data_inputs2 = pickle.load(f)
f.close()
features_STDs = numpy.std(data_inputs2, axis=0)
data_inputs = data_inputs2[:, features_STDs>50]


f = open("outputs.pkl", "rb")
data_outputs = pickle.load(f)
f.close()


n_pop = 50
iters = 10

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

result = pso(2.3, 2.3, iters, n_pop, pop_weights_vector, pop_weights_mat, data_inputs, data_outputs)

print(result[1])
plot_pso(result[1],iters)
pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
best_weights = pop_weights_mat [0, :]
acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="sigmoid")
print("Accuracy of the best solution is : ", result[0])

#plot_pso(result[2],iters)

# f = open("weights_"+str(num_generations)+"_iterations_"+str(mutation_percent)+"%_mutation.pkl", "wb")
# pickle.dump(pop_weights_mat, f)
# f.close()


