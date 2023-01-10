from torch.utils.data import dataset
from torchvision import datasets,transforms
train_dataset = datasets.MNIST(
    root="./data/",
    train=True,
    transform=transforms.ToTensor(),
    download=False
)
test_dataset = datasets.MNIST(
    root="./data/",
    train=False,
    transform=transforms.ToTensor(),
    download=False
)
