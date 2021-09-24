import torch
from torch.distributions import Categorical 

class MixtureGaussianOutputs(torch.nn.Module):
    '''
    Output layer for use with a mixture density network. 

    Outputs the parameters of a mixture of gaussians. 
    Note, by default outputs are pre-activation.
    The predicted_parameters function will give outputs
    with appropriate activation functions.
    '''
    def __init__(self, in_features, output_dim, n_mixtures):
        super(MixtureGaussianOutputs, self).__init__()
        self._output_dim = output_dim
        self._n_mixtures = n_mixtures
        self.mu = torch.nn.Linear(in_features, output_dim*n_mixtures)
        self.log_sigma = torch.nn.Linear(in_features, output_dim*n_mixtures)
        self.log_pi = torch.nn.Linear(in_features, output_dim*n_mixtures)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, input):
        return (self.mu(input).view((-1, self._output_dim, self._n_mixtures)), 
                self.log_sigma(input).view((-1, self._output_dim, self._n_mixtures)), 
                self.log_pi(input).view((-1, self._output_dim, self._n_mixtures))
                )

    def predicted_parameters(self, input):
        return (self.mu(input).view((-1, self._output_dim, self._n_mixtures)), 
                torch.exp(self.log_sigma(input)).view((-1, self._output_dim, self._n_mixtures)), 
                self.softmax(self.log_pi(input).view((-1, self._output_dim, self._n_mixtures))))

    def expected_value(self, input):
        mu, sigma, pi = self.predicted_parameters(input)
        return (pi * mu).sum(dim=2)

    def sample(self, input, n_samples):
        with torch.no_grad():
            mu, log_sigma, log_pi = self.forward(input)
            shape = mu.shape
            device = mu.device
            n = mu.flatten().shape[0]
            samples = mu[None, ...].expand(n_samples, *shape).flatten() + \
                      torch.exp(log_sigma[None, ...].expand(n_samples, *shape).flatten()) * torch.randn(n_samples * n).to(device)
            mask = torch.distributions.Multinomial(1, logits=log_pi).sample((n_samples,))
            samples = (samples.view(n_samples, *shape) * mask).sum(dim=-1)
            return samples

class MixtureDensityNetwork(torch.nn.Module):
    def __init__(self, representation, representation_dim, output_dim=1, n_mixtures=10):
        super(MixtureDensityNetwork, self).__init__()
        self.representation = representation
        #print(f"N MIXTURES: {n_mixtures}")
        self.mixture = MixtureGaussianOutputs(representation_dim, output_dim, n_mixtures)
    
    def forward(self, input):
        hidden = self.representation(input)
        return self.mixture(hidden)

    def sample(self, input, n_samples=1):
        with torch.no_grad():
            hidden = self.representation(input)
            return self.mixture.sample(hidden, n_samples)

    def expected_value(self, input):
        hidden = self.representation(input)
        return self.mixture.expected_value(hidden)

    def predicted_parameters(self, input):
        hidden = self.representation(input)
        return self.mixture.predicted_parameters(hidden)

class BernoulliNetwork(torch.nn.Module):
    def __init__(self, representation, representation_dim, output_dim):
        super(BernoulliNetwork, self).__init__()
        self.representation = representation
        self.logits = lambda x: x.view(x.shape[0], -1)#torch.nn.Linear(representation_dim, output_dim)
    
    def forward(self, input):
        hidden = self.representation(input)
        return self.logits(hidden)

    def sample(self, input, n_samples=1):
        with torch.no_grad():
            p = self.predicted_parameters(input)
            u = torch.rand([n_samples] + list(p.shape)).to(p.device)
            return (u > p).float()

    def expected_value(self, input):
        hidden = self.representation(input)
        return torch.sigmoid(self.logits(hidden))

    def predicted_parameters(self, input):
        return self.expected_value(input)

class MultinomialNetwork(torch.nn.Module):
    def __init__(self, representation, representation_dim, output_dim):
        super(MultinomialNetwork, self).__init__()
        self.representation = representation
        self.logits = torch.nn.Linear(representation_dim, output_dim)
        self.output_dim = output_dim
    
    def forward(self, input):
        hidden = self.representation(input)
        return self.logits(hidden)

    def sample(self, input, n_samples=1):
        m = Categorical(logits=self(input))
        with torch.no_grad():
            m = Categorical(logits=self.forward(input))
            s = m.sample((n_samples, ))

            y_onehot = torch.FloatTensor(n_samples, input.shape[0], self.output_dim)
            y_onehot.zero_()
            y_onehot.scatter_(2, s[:,:,None], 1)
            return y_onehot

    def expected_value(self, input):
        return torch.nn.functional.softmax(self(input))

    def predicted_parameters(self, input):
        return self.expected_value(input)


