import numpy as np 
import skfuzzy as fuzz 
import matplotlib.pyplot as plt

class Architecture():
    def __init__(self,input_size,
                 output_size,
                 num_layers, 
                 num_neurons):
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.layers = []
        for _ in range(num_layers):
            weights = np.random.rand(input_size,num_neurons)
            biases = np.zeros((1,num_neurons))
            self.layers.append({'weights':weights,'biases':biases})
            input_size = num_neurons
        
        weights_output = np.random.rand(num_neurons,output_size)
        biases_output = np.zeros((1,output_size))
        self.layers.append({'weights':weights_output,'biases':biases_output})
    
    def forward_(self,inputs):
        output = inputs 
        for layer in self.layers:
            output = self.sigmoid(
                np.dot(output, layer['weights']) + layer['biases'])
        return output 
    
    def sigmoid(self,x):
        return 1/ (1+ np.exp(-x))


class NeuroGenesis:
    def __init__(self, neuro_architecture, max_neurons):
        self.neuro_architecture = neuro_architecture
        self.max_neurons = max_neurons

    def trigger(self):
        if self.neuro_architecture.num_neurons > 1:
            neuron_to_replace = np.random.randint(
                self.neuro_architecture.num_neurons)

            self.neuro_architecture.layers[-2]['weights'][neuron_to_replace, :] = np.random.rand(
                1, self.neuro_architecture.layers[-2]['weights'].shape[1])
            self.neuro_architecture.layers[-2]['biases'][neuron_to_replace] = 0.0


input_size = 10 
output_size = 10
num_layers = 3 
num_neurons = 5 
max_neurons = 10
architecture =  Architecture(input_size,output_size,num_layers,num_neurons)
inputs = np.random.rand(1, input_size)
output = architecture.forward_(inputs)
print("Output before neurogenesis:")
print(output)

neurogenesis = NeuroGenesis(architecture,max_neurons)
neurogenesis.trigger()

output_ = architecture.forward_(inputs)
print("Output after neurogenesis:")
print(output_)

