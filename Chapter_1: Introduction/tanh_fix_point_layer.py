import torch
import torch.nn as nn

class TanhFixexPointLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(in_features=out_features, out_features=out_features,bias=False)
        self.tol = tol 
        self.max_iter = max_iter 

    def forward(self, x):
        # initialize output to be zero
        z = torch.zeros_like(x)
        self.iterations = 0

        # iterate until convert
        while self.iterations < self.max_iter:
            z_next = torch.tanh(self.linear(z) + x)
            self.err = torch.norm(z_next - z)
            z = z_next
            self.iterations += 1 
            if self.err < self.tol:
                break
        return z 

layer = TanhFixexPointLayer(out_features=50)
print(layer.linear.weight)
X = torch.rand(size=(10,50))
Z = layer.forward(X)
print("Terminate after {} with error {}".format(layer.iteration, layer.err))
print(layer.linear.weight)