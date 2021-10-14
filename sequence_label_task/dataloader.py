import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import numpy as np


class Mydata(Dataset):
    def __init__(self):