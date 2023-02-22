import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

torch.manual_seed(42); # seed rng for reproducibility
device = "mps"

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.c1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.c2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.b2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        self.c3 = nn.Conv2d(16, 16, 3, stride=1, padding=2)
        self.b3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.c1 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=2, output_padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.c2 = nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1 )
        self.b2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        self.c3 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        self.b3 = nn.BatchNorm2d(1)
        self.relu3 = nn.ReLU()

    def forward(self, x):

        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.relu3(x)

        return x

class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def eval(model: Autoencoder, data_loader):
    with torch.no_grad():
        criterion = nn.MSELoss()
        losses = []
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images).cpu()
            losses.append(loss)
    return np.array(losses).mean()

def plot_reconstructed_images(images, model):
    # save few images before and after autoencoder
    outputs = model(images)
    images = images.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, outputs], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
    # save the plot
    plt.savefig(f"autoencoder_reconsturcion.png", bbox_inches='tight')

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 1024

    # Loading MNIST dataset
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    
    images, labels = next(iter(train_loader))

    ae = Autoencoder()
    ae.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    # Train the model
    n_epochs = 30
    for epoch in range(1, n_epochs+1):

        print(f"Epoch {epoch}/{n_epochs}:")
        for images, _ in tqdm(train_loader):
            images = images.to(device)
            # Forward pass
            outputs = ae(images)
            loss = criterion(outputs, images)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch}/{n_epochs}], Train loss: {loss.item():.4f}")
    
    ae.eval()
    eval_loss = eval(ae, test_loader)
    print(f"Eval loss: {eval_loss:.4f}")
    # get a random batch from eval set and plot the reconstructed images
    images, _ = next(iter(test_loader))
    plot_reconstructed_images(images.to(device), ae)

    torch.save(ae.state_dict(), "model/autoencoder.pt")
