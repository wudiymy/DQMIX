import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU


class DRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions * args.n_atom)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        bs = inputs.size(0)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        action_value = F.softmax(self.fc2(h).view(bs, self.args.n_actions, self.args.n_atom), dim=2) #[bs, n_action, n_atom]

        return action_value, h
