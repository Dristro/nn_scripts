from torchvision import transforms, datasets
from torch.utils.data import DataLoader

__all__ = ["load_data", ]

def load_data(train_dir: str,
              test_dir: str,
              transform: transforms.Compose,
              num_workers: int,
              batch_size: int = 32):
    """
    Creates (PyTorch) training and testing dataloaders using the data from given path.

    Args:
        train_dir       - path to the training data
        test_dir        - path to the testing data
        transform       - transforms applied to the data (eg: normalize, resize...)
        num_workers     - number of workers per dataloader
        batch_size      - samples per batch in the dataloader

    Returns:
        A tuple with the train/test dataloaders and the class names
        (train_dataloader, test_dataloader, class_names)
    """
    import torch
    train_data = datasets.ImageFolder(train_dir, transform = transform)
    test_data = datasets.ImageFolder(test_dir, transform = transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = num_workers,
                                  pin_memory = True)
    test_dataloader = DataLoader(test_data,
                                 batch_size = batch_size,
                                 shuffle = True,
                                 num_workers = num_workers,
                                 pin_memory = True)
    return train_dataloader, test_dataloader, class_names