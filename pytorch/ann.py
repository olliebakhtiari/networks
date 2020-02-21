""" Python-based scientific computing package targeted at two sets of audiences:

    - A replacement for NumPy to use the power of GPUs
    - A deep learning research platform that provides maximum flexibility and speed

    Tensors can also be used on a GPU to accelerate computing.

    The 'autograd' package provides automatic differentiation for all operations on Tensors.

    Tensor and Function are interconnected and build up an acyclic graph, that encodes a complete history of
    computation.

    To compute the derivatives, you can call .backward() on a Tensor.

    torch.autograd is an engine for computing vector-Jacobian product.

    nn depends on autograd to define models and differentiate them.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.autograd as autograd         # computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor                  # tensor node in the computation graph


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    num_workers=2,
)
test_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=4,
    shuffle=False,
    num_workers=2,
)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def im_show(img):
    img = img / 2 + 0.5     # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# show images
# im_show(torchvision.utils.make_grid(images))

# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    @classmethod
    def num_flat_features(cls, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    @staticmethod
    def check_cuda_available():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    print(Net.check_cuda_available())

    net = Net()

    # Training dataset.
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            net.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = net.criterion(outputs, labels)
            loss.backward()
            net.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Test dataset.
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # Assess which classes did well.
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    # print('Finished Training')
    # # print(net)
    # params = list(net.parameters())
    # # print(len(params))
    # # print(params[0].size())
    # input_ = torch.randn(1, 1, 32, 32)
    # out = net(input_)
    # # print(out)
    # net.zero_grad()
    # out.backward(torch.randn(1, 10))
    #
    # output = net(input_)
    # target = torch.randn(10)  # a dummy target, for example
    # target = target.view(1, -1)  # make it the same shape as output
    # criterion = nn.MSELoss()
    #
    # loss = criterion(output, target)
    # print(loss)
    #
    # print(loss.grad_fn)  # MSELoss
    # print(loss.grad_fn.next_functions[0][0])  # Linear
    # print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
    #
    # net.zero_grad()  # zeroes the gradient buffers of all parameters
    #
    # print('conv1.bias.grad before backward')
    # print(net.conv1.bias.grad)
    #
    # loss.backward()
    #
    # print('conv1.bias.grad after backward')
    # print(net.conv1.bias.grad)
    #
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # optimizer.zero_grad()  # zero the gradient buffers
    # output = net(input_)
    # loss = criterion(output, target)
    # loss.backward()
    # optimizer.step()

    # x = torch.rand(5, 3, dtype=torch.double)
    # y = torch.rand(5, 3, dtype=torch.double)
    # # print(x)
    # # print(torch.add(x, y))

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")  # a CUDA device object
    #     y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    #     x = x.to(device)  # or just use strings ``.to("cuda")``
    #     z = x + y
    #     print(z)
    #     print(z.to("cpu", torch.double))

    # z = torch.ones(2, 2, requires_grad=True)
    # print(z)
    # a = z + 3
    # print(a)
    # b = a * z * z * 8
    # out = b.mean()
    # # print(out)
    # out.backward()
    # # print(out.grad)

    # x = torch.randn(3, requires_grad=True)
    # y = x * 2
    # while y.data.norm() < 1000:
    #     y = y * 2
    # print(y)

    # v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    # y.backward(v)
    # print(x.grad)
