# Forward Pass a single neuron 

import random 
from main import Value

class Neuron: 
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def parameters(self):
        return self.w + [self.b]
    
    # Call returns w*x+b for a tensor x for a neuron of nin inputs
    def __call__ (self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # pair w and x
        out = act.tanh()
        return out
    
# Example
# x = [2.0, 3.0]
# n = Neuron(2)
# n(x)


# Building a layer of neurons
class Layer:
    def __init__(self, nin, nout):     # nout --> number of neurons we want in the layer
        self.neurons = [Neuron(nin) for i in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()] 
        

# Example
# x = [2.0, 3.0]
# n = Layer(2, 3)
# n(x)


# MLP: Multi Layer Perceptron
class MLP:
    def __init__(self, nin, nouts):  # nouts --> A list of nout sizes of all layers in the MLP
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()] 

# Example
# x = [2.0, 3.0, -1.0]
# n = MLP(3, [4, 4, 1])
# n(x)