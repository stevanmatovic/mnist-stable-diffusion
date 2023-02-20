import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.c1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.c2 = nn.Conv2d(16, 8, 3, stride=1, padding=2)
        self.b2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.c1 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=2, output_padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.c2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        self.b2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)

        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    device = torch.device("mps")

    # Hyperparameters
    batch_size = 1024

    # Loading MNIST dataset
    data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

    images, labels = next(iter(data_loader))

    ae = Autoencoder(Encoder(), Decoder())
    ae.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    # Train the model
    n_epochs = 20
    for epoch in range(n_epochs):

        print(f"Epoch {epoch+1}/{n_epochs}:")
        for images, _ in tqdm(data_loader):

            images = images.to(device)

            # Forward pass
            outputs = ae(images)
            loss = criterion(outputs, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
