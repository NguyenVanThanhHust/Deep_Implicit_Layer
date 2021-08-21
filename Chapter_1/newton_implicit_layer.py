import torch
import torch.nn as nn

class TanhNewtonImplicitLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(in_features=out_features, out_features=out_features,bias=False)
        self.tol = tol 
        self.max_iter = max_iter 

    def forward(self, x):
        # initialize output to be zero
        with torch.no_grad():
            z = torch.zeros_like(x)
            self.iterations = 0

            # iterate until convert
            while self.iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break
                
                # newton step
                J = torch.eye(z.shape[1])[None,:,:] - (1 / torch.cosh(z_linear)**2)[:,:,None]*self.linear.weight[None,:,:]
                z = z - torch.solve(g[:,:,None], J)[0][:,:,0]

                self.iterations += 1 

        # reengage autograd and add the gradient hook
        z = torch.tanh(self.linear(z) + x)
        z.register_hook(lambda grad : torch.solve(grad[:,:,None], J.transpose(1,2))[0][:,:,0])
        return z 

from torch.autograd import gradcheck

layer = TanhNewtonImplicitLayer(5, tol=1e-10).double()
print(gradcheck(layer, torch.randn(3, 5, requires_grad=True, dtype=torch.double), check_undefined_grad=False))
