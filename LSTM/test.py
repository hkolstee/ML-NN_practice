import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# target = torch.randint(0, 10, (10,))
# one_hot = nn.functional.one_hot(target)
# reversed = torch.argmax(one_hot, dim=1)
# print(target)
# print(one_hot)
# print(reversed)

labels = torch.tensor([1, 4, 1, 0, 5, 2])
labels = labels.unsqueeze(0)
target = torch.zeros(labels.size(0), 15).scatter_(1, labels, 1.)
print(target)
