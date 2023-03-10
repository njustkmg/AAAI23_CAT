import mindspore.nn as nn
from misc.utils import *

class MlpHead(Cell):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = Sequential(
            Dense(self.input_dim, self.hidden_dim),
            ReLU(),
            Dense(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        # x shape: [b t d] where t means the number of views
        x = self.model(x)
        return F.normalize(x, dim=-1)
