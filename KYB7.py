import torch
import torch.nn
x = torch.tensor(data:1., requires_grad=True)
w = torch.tensor(data:2., requires_grad=True)
b = torch.tensor(data:3., requires_grad=True)

Y = x*w + b_# Y = a + b

y.backward()

print(x.grad, w.grad, b.grad)


x = torch.randn(10, 3)
y = torch.randn(10, 2)

Linear = nn.linear(in_features:3,out_features:2)
print('w: ', Linear.weight)
print('b: ', Linear.bias)


criterion = nn.MSELoss() # Mean Square Error Loss
optimizer = torch.optim.SGD(Linear.parameters(),lr=0.01)


pred = Linear(x)

loss = criterion(pred, y)
print('loss : ', loss.item())

loss.backward()

    print('dL/dw:', linear.weight.grad)
    print('dL/db:', linear.bias.grad)

x = np.array([[1,2],[3,4]])
y = torch.from_numpy(x)
z = y.numpy()
print("A")
train_dataset = torchvision.datasets.CIFAR10
image.numpy()[0]