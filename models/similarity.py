import torch
import torch.nn as nn

eps = torch.finfo(torch.float).eps

class Simple1DInputNet(nn.Module):
    def __init__(self, config, capacity=64):
        super(Simple1DInputNet, self).__init__()
        self.config = config

        module_list = [
            nn.Linear(self.config["input_channel_count"], capacity),
            nn.ReLU()
        ]
        for i in range(5):
            module_list.append(nn.Linear(capacity, capacity))
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(capacity, 1))
        self.net = nn.Sequential(*module_list)

    def forward(self, x):
        yb = self.net(x)
        yb = yb.squeeze()
        return yb

    def compute_grads(self, x, return_pred=False):
        yb = self.net(x)
        y = yb[0, :]  # Remove batch dim
        # Propagate values of yb separately
        d = 1
        all_dim_grads = []
        for i in range(d):
            self.zero_grad()  # Start over
            y[i].backward(retain_graph=True)
            grads = []
            for tensor in self.parameters():
                grad_flat = tensor.grad.view(-1)
                grads.append(grad_flat)
            grads = torch.cat(grads)
            all_dim_grads.append(grads)
        all_dim_grads = torch.stack(all_dim_grads, dim=1)
        if return_pred:
            return all_dim_grads, y
        else:
            return all_dim_grads


# pytorch can only compute the sum of gradients for each item in the minibatch
# so vectorizing this code is not possible without implementing a custom backwards pass
# see https://github.com/fKunstner/fast-individual-gradients-with-autodiff/tree/master/pytorch
# for a few different methods to do this
def compute_grads(f, x, return_output=False):
    """

    :param f: the NN to use for similarity computation
    :param x: a batch N of inputs to compute grads for
    :param return_output: return the outputs of the network
    :return: (N, D, P) where
    N is the number of inputs, D is the output dimension of the network, P is the total number of network parameters
    """
    y = f(x)
    batch = []
    for b in range(y.size(0)):
        all_dim_grads = []
        for i in range(y.size(1)):
            f.zero_grad()
            y[b, i].backward(retain_graph=True)
            grads = torch.cat([t.grad.flatten(start_dim=0) for t in f.parameters()])
            all_dim_grads.append(grads)

        batch.append(torch.stack(all_dim_grads))
    batch = torch.stack(batch)
    if return_output:
        return batch, y
    else:
        return batch


def similarity_inner_product(f, x, y):
    """
    Computes the unnormalized distance between 2 batches of samples
    :param f: the NN to use for similarity comparison
    :param x: a batch of N samples
    :param y: a batch of N samples to compare
    :return: a matrix of D x D where D is the output dimension of the network
    """
    grad_x = compute_grads(f, x)
    grad_y = compute_grads(f, y)
    return torch.bmm(grad_x, grad_y.permute(0, 2, 1))


def similarity_normed_inner_product(f, x, y):
    """
    Computes the unnormalized distance between 2 batches of samples
    :param f: the NN to use for similarity comparison
    :param x: a batch of N samples
    :param y: a batch of N samples to compare
    :return: N, D, D where D x D is a normalized correlation matrix, D = output dimension of the NN
    """
    grad_x = compute_grads(f, x)
    grad_y = compute_grads(f, y)
    normed_x = grad_x / (grad_x.norm(dim=2, keepdim=True) + eps)
    normed_y = grad_y / (grad_y.norm(dim=2, keepdim=True) + eps)
    return torch.bmm(normed_x, normed_y.permute(0, 2, 1))


def similarity_trace_mean(f, x, y):
    kernel_matrix = similarity_normed_inner_product(f, x, y)
    kernel_matrix_diag = kernel_matrix.diagonal(dim1=1, dim2=2)
    return torch.mean(kernel_matrix_diag, dim=1)