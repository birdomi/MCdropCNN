import json
import torchvision
import torchvision.datasets
import torch
import os

import numpy as np
import PIL

from PIL import Image
from torchvision import transforms
import torch.utils.data as data

imageNet_path = '/data/sung/dataset/imagenet'

train_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]) 
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]) 
])


train_transforms_no_norm = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor()
])

test_transforms_no_norm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class DomainDataset(data.Dataset):
    def __init__(self, path, txt, mode, transform = None):
        assert mode == 'A' or mode == 'B'

        self.transform = transform
        self.domain = txt.split('_')[0]
        self.folder = path
        self.path = os.path.join(
            path,
            self.domain
        )
        if mode == 'A':
            self.label_range = list(range(0, 173))
        if mode == 'B':
            self.label_range = list(range(173, 345))        


        self.items = []
        with open(os.path.join(path, txt)) as f:
            for line in f.readlines():
                img, label = line.split(' ')
                label = int(label.replace('\n', ''))
                if label in self.label_range:
                    self.items.append([self.folder+'/'+img, label])

        self.__find_classes()
        
    def __len__(self):
        return len(self.items)

    def __find_classes(self):
        self.class_to_idx = {cls_idx:i for i, cls_idx in enumerate(self.label_range)}

    def __getitem__(self, index):
        img_path, label = self.items[index]
        #print(img_path, label)

        sample = Image.open(img_path)
        target = self.class_to_idx[label]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

def get_loader_domainNet(folder, domain, mode, batch_size, transform=False):
    assert domain == 'real' or domain == 'quickdraw' or domain =='infograph' or domain == 'sketch'

    train_txt = domain + '_train.txt'    
    test_txt = domain + '_test.txt'

    if transform != None:
        trainset = DomainDataset(folder, train_txt, mode, transform=train_transforms)
        testset = DomainDataset(folder, test_txt, mode, transform=test_transforms)
    else:
        print('no_normalization')
        trainset = DomainDataset(folder, train_txt, mode, transform=train_transforms_no_norm)
        testset = DomainDataset(folder, test_txt, mode, transform=test_transforms_no_norm)


    print(len(trainset), len(testset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory = True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory = True, num_workers = 4)
    return train_loader, valid_loader

def get_loader_imageNet(batch_size):
    imageNet_trainset = torchvision.datasets.ImageFolder(imageNet_path+'/train/', transform = train_transforms)
    imageNet_testset = torchvision.datasets.ImageFolder(imageNet_path+'/val/', transform = test_transforms)
    train_loader = torch.utils.data.DataLoader(imageNet_trainset, batch_size, shuffle=True, pin_memory = True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(imageNet_testset, batch_size, shuffle=False, pin_memory = True, num_workers = 4)
    return train_loader, valid_loader 

def get_loader_domainNet_with_imagNet(folder, batch_size):
    train_txt = 'real_train.txt'
    test_txt = 'real_train.txt'
    trainset = DomainDataset(folder, train_txt, 'A', transform=train_transforms)
    testset = DomainDataset(folder, test_txt, 'A', transform=test_transforms)

    imageNet_trainset = torchvision.datasets.ImageFolder(imageNet_path+'/train/', transform = train_transforms)
    imageNet_testset = torchvision.datasets.ImageFolder(imageNet_path+'/val/', transform = test_transforms)

    new_samples = []
    new_samples.extend(trainset.items)
    for sample in imageNet_trainset.samples:
        new_samples.append([sample[0], sample[1]+len(trainset.class_to_idx)])
        #print(new_samples[-1])
    imageNet_trainset.samples = new_samples
    new_samples = [] 
    new_samples.extend(testset.items)
    for sample in imageNet_testset.samples:
        new_samples.append([sample[0], sample[1]+len(testset.class_to_idx)])
        #print(new_samples[-1])
    imageNet_testset.samples = new_samples        

    print(len(trainset), len(testset), len(imageNet_trainset), len(imageNet_testset), len(imageNet_trainset.classes))
    train_loader = torch.utils.data.DataLoader(imageNet_trainset, batch_size, shuffle=True, pin_memory = True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(imageNet_testset, batch_size, shuffle=False, pin_memory = True, num_workers = 4)
    return train_loader, valid_loader 

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
    
def get_loader_folder(folder, class_range, batch_size=32):
    trainset = torchvision.datasets.ImageFolder(folder+'/train/', transform = train_transforms)
    testset = torchvision.datasets.ImageFolder(folder+'/val/', transform = test_transforms)

    new_samples = []
    for sample in trainset.samples:
        if sample[1] in class_range:
            new_samples.append(sample)
    trainset.samples = new_samples

    new_samples = []
    for sample in testset.samples:
        if sample[1] in class_range:
            new_samples.append(sample)
    testset.samples = new_samples

    print(len(trainset), len(testset))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    return train_loader, valid_loader

def validation_accuracy(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            #print(outputs.shape)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy

if __name__ == '__main__':
    get_loader_domainNet_with_imagNet('/home/lecun/Workspace/yyg/data/domainnet', 32)
    pass
    # import matplotlib.pyplot as plt
    # conf = read_conf('conf/dataset/food101_student.json')
    # a, b = get_loader_with_classlist_bbox_aug('D:/food-101/', 1, conf['id'])

    # for x,y in a:
    #     x = x.squeeze(0).permute([1,2,0]).numpy()
    #     plt.imshow(x)
    #     plt.show()

    # # 
    # # print(conf['id'])
    # # a, b = get_loader_with_classlist('D:/food-101/', 32, conf['id'])
    # # print(len(a.dataset.samples))

    # # a, b = get_loader_with_classlist_teacher('D:/food-101/', 32, conf['id'])
    # # print(len(a.dataset.samples))
