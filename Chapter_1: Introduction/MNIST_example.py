import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.notebook import tqdm 

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

mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=mnist_test, batch_size=100, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
model = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(784, 100), 
    TanhFixexPointLayer(100, max_iter=200),
    nn.Linear(100, 10)
).to(device)

opt = optim.SGD(model.parameters(), lr=1e-1)

def epoch(loader, model, opt=None, monitor=None):
    total_loss, total_err, total_monitor = 0.0, 0.0, 0.0
    model.eval() if opt is None else model.train()
    for idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            if sum(torch.sum(torch.isnan(p.grad)) for p in model.parameters())==0:
                opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if monitor is not None:
            total_monitor += monitor(model)
    return total_err/len(loader.dataset), total_loss/len(loader.dataset), total_monitor/len(loader.dataset)


for i in range(10):
    print("epoch {}".format(i))
    if i == 5:
        opt.param_groups[0]["lr"] = 1e-2

    train_err, train_loss, train_fpiter = epoch(train_loader, model, opt, lambda x : x[2].iterations)
    test_err, test_loss, test_fpiter = epoch(test_loader, model, monitor = lambda x : x[2].iterations)
    print(f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f}, FP Iters: {train_fpiter:.2f} | " +
          f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}, FP Iters: {test_fpiter:.2f}")
