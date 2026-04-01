import torch
import torch.optim as optim
from model import ConvVAE
from utils import get_dataloaders, loss_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loader, _ = get_dataloaders(normal_class=0) # Class 0 = T-shirts

def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device) # No .view(-1, 784) here!
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader.dataset):.4f}')

for epoch in range(1, 21): # Training longer (20 epochs) helps FashionMNIST
    train(epoch)

torch.save(model.state_dict(), "conv_vae_fmnist.pth")