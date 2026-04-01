import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from model import ConvVAE
from utils import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE(latent_dim=128).to(device)
model.load_state_dict(torch.load("conv_vae_fmnist.pth"))
model.eval()

# Normal class was 0 (T-shirt)
NORMAL_CLASS = 0
_, test_loader = get_dataloaders(normal_class=NORMAL_CLASS)

def run_detection():
    recon_errors = []
    labels = [] # 0 for Normal, 1 for Anomaly

    print("Calculating reconstruction errors...")
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            
            # Calculate MSE per image
            # We flatten to (batch, 784) to get a single error value per image
            mse = torch.mean((data.view(-1, 784) - recon.view(-1, 784))**2, dim=1)
            
            recon_errors.extend(mse.cpu().numpy())
            # Map labels: T-shirt -> 0, Everything else -> 1
            binary_labels = [0 if l == NORMAL_CLASS else 1 for l in target]
            labels.extend(binary_labels)

    # Calculate AUROC
    auroc = roc_auc_score(labels, recon_errors)
    print(f"\n>>> Final AUROC Score: {auroc:.4f}")

    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(labels, recon_errors)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC = {auroc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot 2: Error Distribution (This explains the 0.52 score)
    plt.subplot(1, 2, 2)
    recon_errors = np.array(recon_errors)
    labels = np.array(labels)
    plt.hist(recon_errors[labels==0], bins=50, alpha=0.5, label='Normal (T-shirt)', density=True)
    plt.hist(recon_errors[labels==1], bins=50, alpha=0.5, label='Anomaly (Other)', density=True)
    plt.xlabel('Reconstruction Error')
    plt.title('Error Distribution Gap')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_detection()