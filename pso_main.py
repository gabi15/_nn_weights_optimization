import numpy
import ga
import pickle
import ANN
from pso import pso, plot_pso


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
n_pop = 5
iters = 2
c1 = 2.3
c2 = 2.3

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
plot_pso(result[2], iters, n_pop, c1, c2)
print("Accuracy of the best solution is : ", result[1])

# save outputs with weights
# f = open("weights_"+str(iters)+"_iterations_"+str(n_pop)+"_population_c1_"+str(c1)+"_c2"+str(c2)+".pkl", "wb")
# pickle.dump(pop_weights_mat, f)
# f.close()


