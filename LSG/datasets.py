import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


"""# Dataset loading"""

import torchvision.models.resnet

def data_loading(args):
    if args.dataset == 'fashionmnist':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = FashionMNIST(
            './data/fashionmnist_data/', train=True, download=True, transform=trans)
        testset = FashionMNIST(
            './data/fashionmnist_data/', train=False, download=True, transform=trans)
        input_channel = 1  # input color channel num

    elif args.dataset == 'mnist':
        transforms_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = MNIST(
            './data/mnist/', train=True, download=True, transform=transforms_mnist)
        testset = MNIST(
            './data/mnist/', train=False, download=True, transform=transforms_mnist)
        input_channel = 1


    elif args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trainset = CIFAR10(
            './data/cifar/', train=True, download=True, transform=trans)
        testset = CIFAR10(
            './data/cifar/', train=False, download=True, transform=trans)
        input_channel = 3

    #     elif args.dataset == 'ImageNet':
    # Prepare ImageNet beforehand in a different script!
    # We are not going to redownload on every instance
    #         trans = transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])])
    #         trainset = torchvision.datasets.ImageFolder(root='/root/autodl-tmp/imagenet/train',transform=trans)
    #         testset = torchvision.datasets.ImageFolder(root='/root/autodl-tmp/imagenet/val', transform=trans)
    elif args.dataset == 'ImageNet':
        train_transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomResizedCrop(64),  # 对图片尺寸做一个缩放切割
            #         transforms.RandomHorizontalFlip(),  # 水平翻转
            transforms.ToTensor(),  # 转化为张量
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 进行归一化
        ])
        # 对测试集做变换
        val_transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        #         trainset = ImageNet(root='/root/autodl-pub/ImageNet/ILSVRC2012/', split='train',  transform=train_transforms)
        #         testset = ImageNet(root='/root/autodl-tmp/ImageNet/ILSVRC2012/',split='val', transform=val_transforms)
        trainset = ImageFolder(root='D:\BaiduNetdiskDownload\\train\\train', transform=train_transforms)
        testset = ImageFolder(root='D:\BaiduNetdiskDownload\\train\\train', transform=val_transforms)

    elif args.dataset == 'clothing1m':
        data_path = 'D:\数据集\Clothing1M\clothing1M'
        args.num_classes = 14
        args.model = 'resnet50'
        trans_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trans_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trainset = Clothing(data_path, trans_train, "train")
        testset = Clothing(data_path, trans_val, "test")
        n_train = len(trainset)
        # targets = np.array(dataset_train.targets)
    else:
        raise NotImplementedError('Error: unrecognized dataset')

    # Num_classes = len(trainset.classes)  # number of classes in the dataset
    # labels_array = [i for i in range(Num_classes)]  #
    # labels = np.array(trainset.targets)  # retrieve the real labels
    # # noise_or_not = []
    # # img_size = trainset[0][0].shape  # image size
    # classes = np.array(list(trainset.class_to_idx.values()))
    # #     print(len(trainset))
    # #     print(len(testset))
    # #     print(testset[3])
    return trainset, testset


class CIFAR10(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class MNIST(torchvision.datasets.MNIST):
    """Super-class MNIST to return image ids with images."""

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class FashionMNIST(torchvision.datasets.FashionMNIST):
    """Super-class MNIST to return image ids with images."""

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class ImageFolder(torchvision.datasets.ImageFolder):
    """Overwrite torchvision ImageNet to change metafile location if metafile cannot be written due to some reason."""

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, idx) where target is class_index of the target class.

        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class Clothing(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.noisy_labels = {}
        self.clean_labels = {}
        self.data = []
        self.targets = []
        self.transform = transform
        self.mode = mode

        with open(self.root + '\\noisy_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.noisy_labels[img_path] = int(entry[1])

        with open(self.root + '\\clean_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.clean_labels[img_path] = int(entry[1])

        if self.mode == 'train':
            with open(self.root + '\\noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'minitrain':
            with open(self.root + '\\noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            n = len(lines)
            np.random.seed(13)
            subset_idx = np.random.choice(n, int(n/10), replace=False)
            for i in subset_idx:
                l = lines[i]
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'test':
            with open(self.root + '\\clean_test_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.clean_labels[img_path]
                self.targets.append(target)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target, index

    def __len__(self):
        return len(self.data)
