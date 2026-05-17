import torch
import torchvision
import torchvision.transforms as transforms


def loadData(batch_size=100, train_transform=None):
       transform = transforms.Compose(
              [transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
              )
       
       if train_transform is None:
              train_transform = transform
              
       trainset_augmented = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=False, transform=train_transform)

       trainset_original = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=False, transform=transform)

       testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)

       testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

       classes = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
       
       return trainset_augmented, trainset_original, testloader, classes