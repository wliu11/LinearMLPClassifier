from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import csv
import os

"""
Class names for the images
"""
LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


"""
A dataloader that takes the images from a given directory and transforms them each 
into a tensor of size (3, 64, 64), along with the known label of the image.
"""
class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        filepath = dataset_path + '/labels.csv'
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            self.labels = list(reader)

        self.images = list()
        
        for i in range(1, len(self.labels)):
            image_name = self.labels[i]
            image = Image.open(os.path.join(dataset_path + "/" + image_name[0]))
            tensor = transforms.ToTensor().__call__(image).reshape(3, 64, 64)
            self.images.append(tensor)
     
    """
    Returns the length of the overall dataset
    """
    def __len__(self):
        return (len(self.labels) - 1)

    """
    Returns the tuple (image, label)
    """
    def __getitem__(self, idx):
        img = self.images[idx]
        label = LABEL_NAMES.index(self.labels[idx + 1][1])
        return (img, label)

"""
Code provided by the class
"""
def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
