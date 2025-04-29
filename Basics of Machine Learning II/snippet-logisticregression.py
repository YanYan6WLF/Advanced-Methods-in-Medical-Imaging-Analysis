...
import torch as th
from sklearn.model_selection import train_test_split
import numpy as np

# load tensors
input_data = th.load("fruit_data.pt")
labels = th.load("labels.pt")
n_samples=input_data.shape[0]
data_flatten=input_data.reshape((n_samples,-1))
x_train,x_test,y_train,y_test=train_test_split(data_flatten,labels,test_size=0.2,random_state=42)

# specify model
class LogisticRegressionModel(th.nn.Module):
    def __init__(self, D_in, D_out):
        super(LogisticRegressionModel, self).__init__()
        self.linear = th.nn.Linear(D_in, D_out)

    def forward(self, x):
        return th.sigmoid(self.linear(x))

model = LogisticRegressionModel(30000, 1)
loss_fn = th.nn.MSELoss()
optimizer = th.optim.SGD(model.parameters(), lr=1e-3)

loss_values=[]
for t in range(2000):
    def closure():
        # Forward pass: Compute predicted y by passing x to the model
        predicted = model(x_train)

        # Compute and print loss
        loss = loss_fn(predicted, y_train)
        if t % 100 == 99:
            loss_values.append(loss.item())
            print(t,loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        return loss_values

    optimizer.step(closure)


import matplotlib.pyplot as plt

# After the training, plot the convergence plot
plt.figure()
plt.title("Convergence Plot")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.plot(loss_values)
plt.show()

# Evaluate the model to report classification errors
model.eval()  # Set the model to evaluation mode
with th.no_grad():  # Disable gradient calculation
    predicted_train = model(x_train)
    predicted_test = model(x_test)
    train_error = (predicted_train.round() != y_train).sum().item() / len(y_train)
    test_error = (predicted_test.round() != y_test).sum().item() / len(y_test)

print(f"Training classification error: {train_error}")
print(f"Testing classification error: {test_error}")


