<<<<<<< Updated upstream
from torch import argmax, long, manual_seed, tensor, float32
from torch.nn import Sequential, Linear, CrossEntropyLoss, Tanh
=======
from torch import manual_seed, tensor, float32
from torch.nn import Flatten, ReLU, Sequential, Linear, MSELoss, Tanh
>>>>>>> Stashed changes
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD

manual_seed(4)

<<<<<<< Updated upstream
model = Sequential(
    Linear(2, 2, bias=False),
    Tanh(),
    Linear(2, 2, bias=False)
)

X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float32)
y = tensor([0, 1, 1, 0], dtype=long)


loss = CrossEntropyLoss()
optim = SGD(model.parameters(), lr=0.1)
data = DataLoader(TensorDataset(X, y))

for i in range(2000):
=======
model = Sequential(Linear(2, 2, bias=False), ReLU(), Linear(2, 1, bias=False))

X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float32)
y = tensor([[0], [1], [1], [0]], dtype=float32)

loss = MSELoss()
optim = SGD(model.parameters(), lr=0.1)
data = DataLoader(TensorDataset(X, y))

for i in range(500):
>>>>>>> Stashed changes
    for X_b, y_b in data:
        y_pred = model(X_b)
        l = loss(y_pred, y_b)
        l.backward()
        optim.step()
        optim.zero_grad()

print('X\ty\ty_pred')
for X_b, y_b in data:
<<<<<<< Updated upstream
    y_pred = argmax(model(X_b), dim=1)[0]
=======
    y_pred = round(model(X_b).item(), 3)
>>>>>>> Stashed changes
    print(f'{list(map(int, X_b.tolist()[0]))}\t{y_b.tolist()[0]}\t{y_pred}')

layer1 = model[0]
print(f'layer 1\nw = {layer1.weight.tolist()}')

layer2 = model[2]
print(f'layer 2\nw = {layer2.weight.tolist()}')
