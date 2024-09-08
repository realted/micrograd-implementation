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

for k in range(20):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

    # backward pass
    for p in n.parameters():
        p.grad = 0.0 # zero the grad so it doesnt accumulate throughout the iterations
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.05 * p.grad # 0.05 is the learning rate

    print(k, loss.data)