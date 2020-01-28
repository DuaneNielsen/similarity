import torch
import torch.nn as nn
import models.similarity as m
import similarity

x = torch.rand(4, 2)
f = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
grads = similarity.compute_grads(f, x)


def test_comp_grads():
    x = torch.rand(4, 2)
    f = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
    grads = similarity.compute_grads(f, x)
    assert grads.size(0) == 4
    assert grads.size(1) == 2
    assert grads.size(2) == 2


def test_inner_prod():
    x = torch.rand(4, 2)
    f = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1, bias=False))
    sip = similarity.similarity_inner_product(f, x, x)
    print(sip)


def test_normed_inner_prod():
    x = torch.randn(40, 16)
    f = nn.Sequential(nn.Linear(16, 8), nn.Sigmoid(), nn.Linear(8, 4))
    snip = similarity.similarity_normed_inner_product(f, x, x)
    snip = snip.diagonal(dim1=1, dim2=2)
    assert torch.allclose(torch.ones_like(snip), snip)


def test_normed_inner_prod_dist():
    x = torch.randn(40, 16)
    y = torch.randn(40, 16)
    f = nn.Sequential(nn.Linear(16, 8), nn.Sigmoid(), nn.Linear(8, 4))
    snip = similarity.similarity_normed_inner_product(f, x, y)
    print(snip)


def test_trace_mean():
    x = torch.randn(40, 16)
    f = nn.Sequential(nn.Linear(16, 8), nn.Sigmoid(), nn.Linear(8, 4))
    dist = similarity.similarity_trace_mean(f, x, x)
    print(dist)
