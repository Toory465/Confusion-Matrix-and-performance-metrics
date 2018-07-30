from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torch


class CustomDataset(Dataset):
    "Getting set of images and labesl as vectors return items as images"
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images  (numpy array) :
            labels (numpy array) :
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.images = images #pd.read_csv(csv_path)
        self.labels = labels#np.asarray(self.data.iloc[:, 0])
        self.height = 28
        self.width = 28
        self.transform = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = np.asarray(self.images[index])
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)
    #
    def __len__(self):
        return len(self.images)


def load():
    transformations = transforms.Compose([transforms.ToTensor()])
    # load data from cvs file
    data = pd.read_csv('./data/mnist.csv')
    # get labes and data in numpy
    ys, xs = data.values[:, 0], data.values[:, 1:]
    # randomly select 1000 of each label for test set and keep the rest(32000) for train set
    data_set = np.random.permutation(ys[ys == 0].shape[0])
    test_set_inds = data_set[:1000]
    train_set_inds = data_set[1000:]
    for i in range(1, 10):
        data_set = np.random.permutation(ys[ys == i].shape[0])
        test_set_inds = np.concatenate((test_set_inds, data_set[:1000]), axis=0)
        train_set_inds = np.concatenate((train_set_inds, data_set[1000:]), axis=0)

    # reshape test and train data sets
    test_set_images = xs[test_set_inds][:].reshape(-1, 28, 28).astype('uint8')
    test_set_labels = ys[test_set_inds]

    train_set = xs[train_set_inds][:].reshape(-1, 28, 28).astype('uint8')
    train_set_labels = ys[train_set_inds]

    train_data = CustomDataset(train_set, train_set_labels, transformations)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)

    test_data = CustomDataset(test_set_images, test_set_labels, transformations)

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader


