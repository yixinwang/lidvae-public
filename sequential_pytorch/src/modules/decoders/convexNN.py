import sys
import os
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim



def f1(x):
    def f2(y):
        return y*(x-y)
    return torch.autograd.grad(f2(x), x, create_graph=True)[0]

x = torch.tensor(1.0, requires_grad=True)
torch.autograd.grad(f1(x), x)[0]



x = torch.Tensor([1, 1, 1])
x.requires_grad_(True)
y = torch.Tensor([1])
model = torch.nn.Linear(3, 1)
final = torch.nn.Linear(3, 1)


def loss_grad(x):
    def doutdx(x):
        out = model(x)
        print(out)
        dout_dx = torch.autograd.grad(out, [x]+list(model.parameters()), create_graph=True)[0]
        print(dout_dx)
        return dout_dx
    return final(doutdx(x))

out = model(x)
dout_dx = torch.autograd.grad(out, [x], create_graph=True)[0]
final(doutdx(x))

torch.autograd.grad(final(doutdx(x)), model.parameters(), allow_unused=True)


loss_grad(x)

# dout_dx = model.parameters()


loss = (y - (torch.autograd.grad(model(x), [x], create_graph=True)[0]).sum())**2

d2loss = []
for param in model.parameters():
    print(param.grad.data, param)
    # drv = torch.autograd.grad(loss, param, create_graph=True)
    # d2loss.append(drv)
    # print(param, drv)


dloss = torch.autograd.grad(loss, (model.parameters()), create_graph=True)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)


theta = torch.Tensor([1, 2, 3])
theta.requires_grad_(True)
c = (x * theta).sum()
print(c.grad_fn)  # c has gradient function
dc_dx = torch.autograd.grad(c.mean(), [x], create_graph=True)[0]
print(dc_dx)
print(dc_dx.grad_fn)  # derivative of c over x doesn't have gradient function
dc_dx_dtheta = torch.autograd.grad(dc_dx.sum(), [theta], create_graph=True)
dc_dx.backward([theta])  # raises RuntimeError

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.autograd import grad
from torch import nn

# some toy data
x = Variable(Tensor([4., 2.]), requires_grad=False)
y = Variable(Tensor([1.]), requires_grad=False)

# linear model and squared difference loss
model = nn.Linear(2, 1)
loss = torch.sum((y - model(x))**2)

# instead of using loss.backward(), use torch.autograd.grad() to compute gradients
loss_grads = grad(loss, model.parameters(), create_graph=True)

# compute the second order derivative w.r.t. each parameter
d2loss = []
for param, grd in zip(model.parameters(), loss_grads):
  for idx in iterator_over_tensor(param)
    drv = grad(grd[idx], param[idx], create_graph=True)
    d2loss.append(drv)
    print(param, drv)

    
# class ICNN():
# 	def __init__(self):
# 		self.


# def icnn_grad(self, x_train_tensor):
#     with tf.GradientTape() as icnn_tape:
#         icnn_tape.watch(x_train_tensor)
#         h = [[None] for i in range(self.px_num_hidden_layers + 1)]
#         h[0] = tf.square(tf.nn.leaky_relu(self.Wy0(x_train_tensor)))
#         for i in range(self.px_num_hidden_layers):
#             h[i+1] = tf.nn.leaky_relu(self.icnn_Wz_layers[i](h[i]) + self.icnn_Wy_layers[i](x_train_tensor))
#     dout_dx = icnn_tape.gradient(h[-1], x_train_tensor)
#     return dout_dx