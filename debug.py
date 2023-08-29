import torch

def fn(input):
    return torch.clamp(torch.tensor([True]), torch.tensor([True]), input)

x = torch.rand([1], requires_grad=True) # works fine if requires_grad=False

ret_eager = fn(x)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
ret_compiled = compiled(x)
print('==== torchcomp mode OK! ====')
