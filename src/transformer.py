from torchvision import transforms

class TrainTransforms:
    """
    Data augmentation and preprocessing class for training data.
    """
    def __init__(self, mu=0.5, st=0.5):
        self.mu = mu
        self.st = st
        self.transforms = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),  # Random resized crop to 48x48
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),  # Random color jitter
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),  # Random translation
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),  # Random rotation
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=(self.mu,), std=(self.st,)),  # Normalize
            transforms.RandomErasing()  # Random erasing
        ])

    def __call__(self, img):
        return self.transforms(img)


class ValidTransforms:
    """
    Data preprocessing class for validation data.
    """
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.ToTensor()  # Convert to tensor
        ])

    def __call__(self, img):
        return self.transforms(img)


