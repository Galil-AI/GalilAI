""" Probabilistic Neural Network
    outputs means and variances given an input which define a 
    distribution from which the next state can be sampled
"""
import numpy as np 
import torch 

CUDA0 = 'cuda:0'
CPU = 'cpu'

class PNN:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, seed=None):
        if seed is not None:
            torch.manual_seed(seed) if seed is not None else print()
        else: 
            print('warning: random seed not specified')
        self.seed = seed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.output_dim = 2 * state_dim  ## predict (mean,var) for each state dim (assuming independence)
        self.hidden_units = 512  ## #32 #512
        self.device = torch.device(CUDA0 if torch.cuda.is_available() else CPU)

        # Instantiate model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_units, bias=True),
            torch.nn.Linear(self.hidden_units, self.hidden_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units, self.hidden_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units, self.output_dim, bias=True)
        ).to(self.device)

        # Instantiate optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.999

    def softplus(self, x):
        """ Compute softplus """
        softplus = torch.log(1 + torch.exp(x))
        # Avoid infinities due to taking the exponent
        softplus = torch.where(softplus==float('inf'), x, softplus)
        return softplus

    def NLL(self, mean, var, truth):
        """ Compute the negative log likelihood """
        diff = torch.sub(truth, mean)
        var = self.softplus(var)
        loss = torch.div(torch.square(diff), 2 * var)
        loss += 0.5 * torch.log(var)    # + np.log(2 * np.pi)
        loss = torch.sum(loss, dim=1)
        return loss.mean()

    def forward(self, X):
        """ Forward pass for a given input
            :X:       (state,action)-pair
            :returns: means and variances given input
        """
        # Compute output of model
        if isinstance(X, np.ndarray):
            x = torch.from_numpy(X).float().to(self.device)
            out = self.model(x)
            mean, var = torch.split(out, self.output_dim//2, dim=1)
            var = self.softplus(var)
            mean, var = mean.cpu(), var.cpu()
            return mean.detach().numpy(), var.detach().numpy()
        elif isinstance(X, torch.Tensor):
            out = self.model(X.to(self.device))
            mean, var = torch.split(out, self.output_dim//2, dim=1)
            var = self.softplus(var)
            return mean, var
        else:
            raise NotImplementedError
        
    def step(self, inputs, true_out, adjust_lr=False):
        """ Execute gradient step given the samples in the minibatch """
        # Convert input and true_out to useable tensors
        X = torch.from_numpy(inputs).float().to(self.device)
        y = torch.from_numpy(true_out).float().to(self.device)

        # Compute output of model
        out = self.model(X)
        # print(out.shape)

        mean, var = torch.split(out, self.output_dim//2, dim=1)
        # print(mean.shape, var.shape)

        # Compute loss 
        self.nll = self.NLL(mean, var, y)

        # Backpropagate the loss
        self.optimizer.zero_grad()
        self.nll.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        if adjust_lr:
            self.adjust_learning_rate()

    def compute_errors(self, train_X, train_y, test_X=None, test_y=None):
        """ Compute loss on the training, validation and test data """
        # Training data
        train_X = torch.from_numpy(train_X).float().to(self.device)
        train_y = torch.from_numpy(train_y).float().to(self.device)
        mean, var = self.forward(train_X)
        train_loss = self.NLL(mean, var, train_y).item()

        # Test data
        if test_X is not None:
            test_X = torch.from_numpy(test_X).float().to(self.device)
            test_y = torch.from_numpy(test_y).float().to(self.device)
            mean, var = self.forward(test_X)
            test_loss = self.NLL(mean, var, test_y).item()
        else:
            test_loss = None

        return train_loss, test_loss
