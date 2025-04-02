import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


# 搭建神经网络
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.model = torch.nn.Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(64 * 4 * 4, 64),
            Linear(64, 10),  # CIFAR10有10个类别
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = MyModule()
    print(model)

    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print(output.shape)
