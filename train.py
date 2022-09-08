from model import LinearNet
import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)



n_epochs = 20
learning_rate = 0.01

model = LinearNet(10)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs):
    y_pred = model(X)

    l = loss(Y, y_pred)

    l.backward()

    optimizer.step()

    if epoch % 10  == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item( ):.3f}, loss = {l:.8f}")

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')