import math
import torch
import torch.nn as nn
from torch.nn import Parameter


class ClusterAssignment(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            embedding_dimension: int,
            alpha: float = 1.0,
            device=torch.device('cuda')
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        initial_cluster_centers = torch.zeros(
            self.cluster_number, self.embedding_dimension, dtype=torch.float, device=device
        )
        nn.init.xavier_uniform_(initial_cluster_centers)

        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [sample sizes N, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class GCNskip(nn.Module):
    """GCN层"""

    def __init__(self, input_features, output_features, bias=False, activation=nn.ReLU(), device=torch.device('cuda')):
        super(GCNskip, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self._activation = activation
        self.weights_g = nn.Parameter(torch.rand(input_features, output_features, device=device))
        self.weights_l = nn.Parameter(torch.rand(input_features, output_features, device=device))
        if bias:
            self.bias = nn.Parameter(torch.rand(output_features, device=device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        std = 1. / math.sqrt(self.weights_g.size(1))
        self.weights_g.data.uniform_(-std, std)
        self.weights_l.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input):
        (adj, x) = input
        assert adj.shape[0] == adj.shape[1], "GCN input error！"
        support = torch.mm(x, self.weights_g)
        output1 = torch.mm(adj, support)
        output2 = torch.mm(x, self.weights_l)
        output = output1 + output2
        if self.bias is not None:
            return output + self.bias
        if self._activation is not None:
            output = self._activation(output)
        return (adj, output)


class GCNLayer(nn.Module):
    """GCN层"""

    def __init__(self, input_features, output_features, bias=False, activation=nn.ReLU(), device=torch.device('cuda')):
        super(GCNLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self._activation = activation
        self.weights = nn.Parameter(torch.rand(input_features, output_features, device=device))
        if bias:
            self.bias = nn.Parameter(torch.rand(output_features, device=device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        std = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input):
        (adj, x) = input
        assert adj.shape[0] == adj.shape[1], "GCN input error！"
        support = torch.mm(x, self.weights)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        if self._activation is not None:
            output = self._activation(output)
        return (adj, output)


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
        #         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)

        return z

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     z = mu.add(std)
    #
    #     return z


class GCNGaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features, device=torch.device('cuda')):
        super(GCNGaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = GCNLayer(in_features, out_features, device=device, activation=None)
        self.log_var = GCNLayer(in_features, out_features, device=device, activation=None)

    def forward(self, adj, x):
        _, mu = self.mu((adj, x))
        _, log_var = self.log_var((adj, x))

        return self.reparametrize(mu, log_var), mu, log_var

    def encode(self, adj, x):
        _, mu = self.mu((adj, x))
        _, log_var = self.log_var((adj, x))
        return mu, mu, log_var


class MLPGaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(MLPGaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.reparametrize(mu, log_var), mu, log_var

    def encode(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, mu, log_var


def build_gcn(layers, activation=nn.ReLU(), bn=False, dropout=0, device=torch.device('cuda'),basic="GCN"):
    """
    Build multilayer linear perceptron
    layers: example:[input_dim, h_dim1, h_dim2, ... z_dim]
    """
    net = []
    for i in range(1, len(layers)):
        if basic == "GCN":
            net.append(GCNLayer(layers[i - 1], layers[i], activation=activation, device=device))
        elif basic == "GCNskip":
            net.append(GCNskip(layers[i - 1], layers[i], activation=activation, device=device))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

class GCNskip(nn.Module):
    """GCN层"""

    def __init__(self, input_features, output_features, bias=False, activation=nn.ReLU(), device=torch.device('cuda')):
        super(GCNskip, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self._activation = activation
        self.weights_g = nn.Parameter(torch.rand(input_features, output_features, device=device))
        self.weights_l = nn.Parameter(torch.rand(input_features, output_features, device=device))
        if bias:
            self.bias = nn.Parameter(torch.rand(output_features, device=device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        std = 1. / math.sqrt(self.weights_g.size(1))
        self.weights_g.data.uniform_(-std, std)
        self.weights_l.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input):
        (adj, x) = input
        assert adj.shape[0] == adj.shape[1], "GCN input error！"
        support = torch.mm(x, self.weights_g)
        output1 = torch.mm(adj, support)
        output2 = torch.mm(x, self.weights_l)
        output = output1 + output2
        if self.bias is not None:
            return output + self.bias
        if self._activation is not None:
            output = self._activation(output)
        return (adj, output)
