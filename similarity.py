import torch
from datasets import package
from torch.utils.data.dataloader import DataLoader
from models import autoencoder, maker
from config import config
from tqdm import tqdm

eps = torch.finfo(torch.float).eps

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


def similarity_trace_mean(f, x, y, return_diag=False):
    kernel_matrix = similarity_normed_inner_product(f, x, y)
    kernel_matrix_diag = kernel_matrix.diagonal(dim1=1, dim2=2)
    trace_mean = torch.mean(kernel_matrix_diag, dim=1)
    if return_diag:
        return trace_mean, kernel_matrix_diag
    else:
        return trace_mean


if __name__ == '__main__':

    args = config()
    torch.cuda.set_device(args.device)

    """ data """
    datapack = package.datasets[args.dataset_name]
    train, test = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.dataroot)
    train_l = DataLoader(train, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    test_l = DataLoader(test, batch_size=args.batchsize, shuffle=True, drop_last=True, pin_memory=True)

    """ model """
    encoder = maker.make_layers(args.model_encoder, type=args.model_type)
    decoder = maker.make_layers(args.model_decoder, type=args.model_type)
    auto_encoder = autoencoder.LinearAutoEncoder(encoder, decoder, init_weights=args.load is None).to(args.device)
    auto_encoder.load_state_dict(torch.load(args.load))

    batch = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for images, labels in tqdm(train_l):
        for image, label in zip(images, labels):
            batch[label.item()].append(image)

    sim_stats = torch.zeros(len(batch), len(batch))

    for numeral_left in tqdm(batch):
        for numeral_right in batch:
            length = min(len(batch[numeral_left]), len(batch[numeral_right]), 50)
            left = torch.stack(batch[numeral_left][0:length]).to(args.device)
            right = torch.stack(batch[numeral_right][0:length]).to(args.device)
            sim_stats[numeral_left, numeral_right] = similarity_trace_mean(auto_encoder.encoder,
                                                                           left.flatten(start_dim=1),
                                                                           right.flatten(start_dim=1)).mean()
    torch.set_printoptions(sci_mode=False)
    print(sim_stats)


