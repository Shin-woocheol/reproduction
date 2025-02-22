import random


class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.batch_size = batch_size

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position % self.capacity] = experience 
        self.position += 1

    def sample(self):
        out = random.sample(self.memory, self.batch_size)
        return list(map(list, zip(*out)))

    def can_sample(self):
        return len(self.memory) >= self.batch_size

    def episode_sample(self):
        out = self.memory
        return list(map(list, zip(*out))) #* *unpack 후, zip을 하면 종류별로 여러 튜플로 묶일 듯. 거기에 list취한 거.

    def __len__(self):
        return len(self.memory)

class RolloutBuffer:
    '''
    general buffer.
    '''
    def __init__(self):
        self.buffer = list()

    def store(self, trainsition):
        self.buffer.append(trainsition)
    
    def sample(self):
        out = self.buffer
        return out

    @property # @property decorator를 붙이면 buffer.size 로 buffer.size()를 이용할 수 있다.
    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()