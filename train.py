from model import NeuralNetwork


n_epochs = 20

model = NeuralNetwork()

for epoch in range(1, n_epochs):
    loss_train = 0.0
    for imgs, numeric_features_, Y in train_loader:
        imgs = imgs.to(device)
        numeric_features = numeric_features.to(device)
        Y = Y.to(device)

        output = model(imgs, numeric_features)

        loss = loss_fn(output, Y)

        # l2 Regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss+l2_lambda*l2_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
