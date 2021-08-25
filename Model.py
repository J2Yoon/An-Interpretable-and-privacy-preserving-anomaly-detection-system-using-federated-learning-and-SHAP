
import torch
import torch.nn as nn



class model(nn.Module):
    def __init__(self, input_size, center_size):
        super(model, self).__init__()
        self.fc1 = nn.Linear(input_size, center_size)
        self.fc2 = nn.Linear(center_size, 2)
        self.active=nn.ReLU()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sequencal = nn.Sequential(self.fc1,self.active, self.fc2)



    def forward(self, x):
        out = self.sequencal(x)
        dec_out=1

        return out, dec_out






