import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler,BatchSampler
from load_dataset import CustomDataset


class CustomBatch(BatchSampler):

  def __init__(self, sampler, batch_size, drop_last, overlap):
    super().__init__(sampler, batch_size, drop_last)

    self.overlap = overlap

  def __iter__(self):
    batch = []
    start = 0

    #print('Overlap = ',self.overlap)
    #print('Batch Size =', self.batch_size)

    for id in range(0, len(list(self.sampler)), self.batch_size-self.overlap ):
      temp = list(self.sampler)
      batch.extend(temp[start: start + self.batch_size])
      start = start + self.batch_size - self.overlap
      if len(batch) == self.batch_size:
        yield batch
        batch = []
      if len(batch) > 0 and not self.drop_last:
        yield batch

  def __len__(self):

    if self.drop_last:
      return len(self.sampler) // self.batch_size
    else:
      return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    




def split_train_val(train_size, image_size, hazy_list, clear_list, batch_size, overlap):
    '''
        Divide the dataset into train/test

    Parameters:
        - train_size (float) : ratio to split train and validation 
        - image_size (int) : size of an input image
        - hazy_list (list,str) :  list of path  hazy images
        - clear_list (list,str) : list of path for clear images
        - batch_size (int) : size of each batch

    Returns:
        - Two generators (train_loader, test_loader)
    '''

    transform = transforms.Compose([
    transforms.Resize((image_size,image_size)) ,
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    indices = list(range(len(hazy_list)))
    split = int(np.floor(train_size*len(hazy_list)))
    train_idx, val_idx = indices[:split], indices[split:]

    train_sampler = CustomBatch(SequentialSampler(train_idx), batch_size, drop_last = True, overlap = overlap)
    test_sampler = CustomBatch(SequentialSampler(val_idx), batch_size, drop_last = True, overlap = overlap)

    train_dataset = CustomDataset(hazy_list, clear_list,transform, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)

    test_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=test_sampler, num_workers=0)

    return train_loader, test_loader