from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

from datasets.loader import OfficeHome

data_root = '/home/wangkai/handwrite/data'
office31_root = '/home/wangkai/handwrite/data/OFFICE31'
office_home_root = '/home/wangkai/handwrite/data/OfficeHome'


class MyLoader(DataLoader):
    def __init__(self, dataset, domain, **kwargs):
        super().__init__(dataset, **kwargs)
        self.domain = domain


def domain_loader(domain, batch_size=32, num_workers=8, train_ratio=0.8, use_transforms=True):
    # 数据增强变换（用于训练集）
    train_transforms = transforms.Compose([
        transforms.Resize([256, 256]),  # 调整大小
        transforms.RandomResizedCrop(224),  # 随机裁剪为224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomRotation(30),  # 随机旋转±30度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.RandomGrayscale(p=0.1),  # 10%的概率转换为灰度图
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 基础变换（用于测试集）
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),  # 调整大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 根据use_transforms参数选择变换
    if use_transforms:
        train_transform = train_transforms
        test_transform = test_transforms
    else:
        train_transform = test_transform = transforms.Compose([
            transforms.Resize([224, 224]),  # 调整大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])

    # 加载指定domain的数据
    train_dataset = OfficeHome(office_home_root, domain, transform=train_transform)
    test_dataset = OfficeHome(office_home_root, domain, transform=test_transform)

    train_size = int(len(train_dataset) * train_ratio)
    test_size = len(train_dataset) - train_size

    train_dataset, _ = random_split(train_dataset, [train_size, test_size])
    _, test_dataset = random_split(test_dataset, [train_size, test_size])

    train_loader = MyLoader(
        train_dataset,
        domain=domain,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = MyLoader(
        test_dataset,
        domain=domain,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    print(f"{domain.capitalize()} - Train: {train_size}, Test: {test_size}")

    return train_loader, test_loader
