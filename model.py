import torch as t
from torch import nn
from collections import namedtuple

Arch = namedtuple('Arch', 'chan1 chan2 fc1 fc2')
arch = Arch(12, 12, 300, 25)

class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.net = nn.Sequential(
        Unsqueeze(1),
        nn.Conv2d(1, arch.chan1, (5, 5), 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 1),
        nn.Conv2d(arch.chan1, arch.chan2, (5, 5), 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 1),
        nn.Flatten(),
        nn.Linear(arch.fc1, arch.fc2),
        nn.Sigmoid(),
        nn.Linear(arch.fc2, 8)
        nn.Sigmoid()
        )

    self.loss = nn.CrossEntropyLoss()

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out',
            nonlinearity='relu')
      elif isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)

  def forward(self, x):
    return self.net(x)

  def save(self, path):
    t.save(self.state_dict(), path)

  def restore(self, path):
    state_dict = t.load(path)
    self.load_state_dict(state_dict, strict=True)


  



