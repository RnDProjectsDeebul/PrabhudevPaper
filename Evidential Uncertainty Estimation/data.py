import torch
from torchvision.datasets.mnist import MNIST
# from torchvision.datasets.FER2013 import FER2013
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# data_train = MNIST("./data/mnist",
#                    download=True,
#                    train=True,
#                    transform=transforms.Compose([transforms.ToTensor()]))

# data_val = MNIST("./data/mnist",
#                  train=False,
#                  download=True,
#                  transform=transforms.Compose([transforms.ToTensor()]))

# dataloader_train = DataLoader(
#     data_train, batch_size=1000, shuffle=True, num_workers=8)
# dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=8)

# dataloaders = {
#     "train": dataloader_train,
#     "val": dataloader_val,
# }
# digit_one, _ = data_val[5]
# print(type(digit_one))




train_data = 'Data/train'
transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.ImageFolder(train_data, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)



test_data = 'Data/val'
transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = datasets.ImageFolder(test_data, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

dataloaders = {
    "train": train_dataloader,
    "val": test_dataloader,
}
# print(type(test_dataset))
for i, (inputs, labels) in enumerate(test_dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    inputs = inputs.to(device)
    labels = labels.to(device)
# images, labels = next(iter(test_dataloader))
    digit_one = inputs
    if i ==1:
        break
# print(type(digit_one))



# train_data = 'LISA/train'

# transform = transforms.Compose([transforms.Resize(224),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# train_dataset = datasets.ImageFolder(train_data, transform=transform)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)


# # print("2")
# test_data = 'LISA/val'

# transform = transforms.Compose([transforms.Resize(224),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# test_dataset = datasets.ImageFolder(test_data, transform=transform)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

# dataloaders = {
#     "train": train_dataloader,
#     "val": test_dataloader,
# }
# print(type(test_dataset))
# for i, (inputs, labels) in enumerate(test_dataloader):
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     inputs = inputs.to(device)
#     labels = labels.to(device)
# # images, labels = next(iter(test_dataloader))
#     digit_one = inputs
#     if i ==1:
#         break
# print(type(digit_one))




# train_data = 'Imagenette/train'

# transform = transforms.Compose([transforms.Resize(224),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# train_dataset = datasets.ImageFolder(train_data, transform=transform)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)


# # print("2")
# test_data = 'Imagenette/val'

# transform = transforms.Compose([transforms.Resize(224),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# test_dataset = datasets.ImageFolder(test_data, transform=transform)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

# dataloaders = {
#     "train": train_dataloader,
#     "val": test_dataloader,
# }
# print(type(test_dataset))
# for i, (inputs, labels) in enumerate(test_dataloader):
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     inputs = inputs.to(device)
#     labels = labels.to(device)
# # images, labels = next(iter(test_dataloader))
#     digit_one = inputs
#     if i ==1:
#         break
# print(type(digit_one))
