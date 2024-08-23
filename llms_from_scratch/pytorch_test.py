import torch
import torch.nn.functional as F

from torch.autograd import grad
from src import module

def test_matrix():
    A = torch.tensor([[1,2,3],[4,5,6]])
    print(A.shape)
    print(A.dtype,A.to(torch.float32))
    print(A @ A.T)

    return

def test_autograd():
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2],requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1 * w1 + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)
    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b  = grad(loss, b, retain_graph=True)
    print(grad_L_w1, grad_L_b)

    print(w1.grad, b.grad)
    loss.backward()
    print(w1.grad, b.grad)
    return


class MultiLayersPerceptionExample(torch.nn.Module):
    def __init__(self,input_dims,output_dims):
        super().__init__()
        self.layer = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(input_dims, 30),
            torch.nn.ReLU(),

            # 2nd
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output
            torch.nn.Linear(20, output_dims),
        )
    def forward(self,x):
        logits = self.layer(x)
        return logits


def test_multilayers():
    torch.manual_seed(123)
    model =  MultiLayersPerceptionExample(50,3)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num_params:',num_params)
    print('weight shape:',model.layer[0].weight.shape)
    print('bias:',model.layer[0].bias)

    X = torch.rand((1,50))
    out = model(X)
    print('out:',out)

    return

if __name__ == '__main__':

    module.print_line('test_matrix', True)
    test_matrix()
    module.print_line('test_matrix', False)

    module.print_line('test_autograd', True)
    test_autograd()
    module.print_line('test_autograd', False)

    module.print_line('test_multilayers', True)
    test_multilayers()
    module.print_line('test_multilayers', False)