import torch
from torch.autograd import Variable

a = torch.Tensor([3])
b = torch.Tensor([4])

x = Variable(a, requires_grad=True)
w = Variable(b, requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

y = w*x + b

y.backward()

print x.grad
print w.grad
print b.grad