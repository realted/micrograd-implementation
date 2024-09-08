from main import Value
from NN import Neuron, Layer, MLP

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)


xs = [
    [2.0, 3.0, -1.0], 
    [3.0, -1.0, 0.5], 
    [0.5, 1.0, 1.0], 
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0] # desired targets 
# Output [1.0] for [2.0, 3.0, -1.0], etc. 
ypred = [n(x) for x in xs]
ypred

loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
loss
# We want loss to be low

loss.backward()
# This backpropagates to all the neurons in the network (its weights, bias, etc.) such that we can determine the grad of the weights for tuning

#print(n.parameters())

#Access individual weights
n.layers[0].neurons[0].w[0].data

# Gradient Descent
# The process of tuning the weights by a small step size (e.g. 0.01)

for p in n.parameters():
    p.data += -0.01 * p.grad # Negative sign to decrease the gradient of loss 

# Forward pass and check loss function (should be less than prev) 

ypred = [n(x) for x in xs]
loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
print(loss)


# iterate backpropagation and gradient descent
for p in n.parameters():
    p.grad = 0.0 # zero the grad so it doesnt accumulate throughout the iterations
loss.backward()
for p in n.parameters():
    p.data += -0.01 * p.grad # Negative sign to decrease the gradient of loss 
    
ypred = [n(x) for x in xs]
loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

print(loss)
# Loss should be less than the previous loss