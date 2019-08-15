from torchvision import transforms


def get_transform():
    mean = [104 / 255.0, 117 / 255.0, 128 / 255.0]
    std = [1.0 / 255, 1.0 / 255, 1.0 / 255]
    normalize = transforms.Normalize(mean, std)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(scale=(0.16, 1), size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    }
    return data_transforms
