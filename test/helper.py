import torch.nn as nn
import torch
import torch.nn.functional as F
import string 

class RNNTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNTorch, self).__init__()
        
        g = torch.Generator().manual_seed(21473647) 
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
        self.i2h.weight.data.uniform_(-0.1, 0.1, generator=g)
        self.i2h.bias.data.uniform_(-0.1, 0.1, generator=g)
        self.h2h.weight.data.uniform_(-0.1, 0.1, generator=g)
        self.h2h.bias.data.uniform_(-0.1, 0.1, generator=g)
        self.h2o.weight.data.uniform_(-0.1, 0.1, generator=g)
        self.h2o.bias.data.uniform_(-0.1, 0.1, generator=g)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

mock_data={
 'Axon': 'English',
 'Pavlyuk': 'Russian',
 'Peck': 'English',
 'Charnetsky': 'Russian',
 'Gerst': 'German',
 'Bergamaschi': 'Italian',
 'Ballantyne': 'English',
 'Seredavin': 'Russian',
 'Beutel': 'German',
 'Glubotsky': 'Russian',
 'Adilov': 'Russian',
 'Arah': 'English',
 'Batchish': 'Russian'
}

all_categories=[
 'Russian','Korean','Polish',
 'Dutch','Chinese','Czech',
 'Greek','Portuguese','English',
 'Vietnamese','Japanese','French',
 'Irish','Scottish','German',
 'Spanish','Italian','Arabic']

all_letters = string.ascii_letters + " .,;'"