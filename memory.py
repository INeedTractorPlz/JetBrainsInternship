from collections import namedtuple
import random
from torch.utils.data import Dataset
import numpy as np

class ReplayDataset(Dataset):

    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.memory = np.array([])
        self.position = 0
        self.Transition = Transition

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory = np.append(self.memory, None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def __getitem__(self, index):
        return self.memory[index]
    
    def __len__(self):
        return len(self.memory)

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = Transition

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)