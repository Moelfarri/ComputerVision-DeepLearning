from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
np.random.seed(0)

mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)

#Class for data augmentation
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_cifar10(batch_size: int, validation_fraction: float = 0.1,
                 task4a=False, addGN=False, chrismodel=False) -> typing.List[torch.utils.data.DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!
    
    #resize image and normalize for task4a/otherwise stay the same:
    if not task4a:
        print("INIT DATA FOR TASK2/TASK3")
        
        #add gaussian noise for data augmentation
        if addGN:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-3,3)),
                AddGaussianNoise(0., 1.) 
                #Maybe random erase would be nice to try too?
            ])
            
        elif chrismodel:
            print("CHRIS MODEL DATALOADING")
            transform_train = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-3,3))
            ])
            
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        print("INIT DATA FOR TASK4a")
        mean_resnet = [0.485, 0.456, 0.406] #standard mean/std for pytorch/imageNet
        std_resnet =  [0.229, 0.224, 0.225]
        
       
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)), #use 112x112 if training takes to long
            transforms.ToTensor(),
            transforms.Normalize(mean_resnet, std_resnet),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)), #use 112x112 if training takes to long
            transforms.ToTensor(),
            transforms.Normalize(mean_resnet, std_resnet),
        ])
    
    
    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    data_test = datasets.CIFAR10('data/cifar10',
                                 train=False,
                                 download=True,
                                 transform=transform_test)

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test
