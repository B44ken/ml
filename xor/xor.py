from torch import tensor
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD

model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1), ReLU())
X = tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = tensor([0., 1., 1., 0.])

data = DataLoader(TensorDataset(X, y))

loss = CrossEntropyLoss()

optim = SGD(model.parameters(), lr=0.0001)

for i in range(100):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in data:
        optim.zero_grad()
        outputs = model(inputs)
        loss_value = loss(outputs, targets.long())
        loss_value.backward()
        optim.step()
        
        running_loss += loss_value.item()
    
    if i % 10 == 0:
        print(f'Epoch {i+1}, Loss: {running_loss}')

model.eval()
with torch.no_grad():
    output = model(tensor([[1., 0.]]))
    print(output)

