from torch import long, manual_seed, tensor, float32
from torch.nn import Sequential, Linear, CrossEntropyLoss, Tanh
from torch import manual_seed, tensor, float32
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD

manual_seed(4)

model = Sequential(
    Linear(2, 2, bias=True),
    Tanh(),
    Linear(2, 2, bias=True),
    Tanh()
)

X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float32)
y = tensor([0, 1, 1, 0], dtype=long)

loss = CrossEntropyLoss()
optim = SGD(model.parameters(), lr=0.1)
data = DataLoader(TensorDataset(X, y))

for i in range(2000):
    for X_b, y_b in data:
        y_pred = model(X_b)
        l = loss(y_pred, y_b)
        l.backward()
        optim.step()
        optim.zero_grad()

print('X\ty\ty_pred')
for X_b, y_b in data:
    y_pred = model(X_b).argmax().item()
    print(f'{list(map(int, X_b.tolist()[0]))}\t{y_b.tolist()[0]}\t{y_pred}')

layer1 = model[0]
print(f'layer 1\nw = {layer1.weight.tolist()}')

layer2 = model[2]
print(f'layer 2\nw = {layer2.weight.tolist()}')
