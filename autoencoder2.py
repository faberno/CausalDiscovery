import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=64
        )
        self.encoder_output_layer = nn.Linear(
            in_features=64, out_features=121
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=121, out_features=64
        )
        self.decoder_output_layer = nn.Linear(
            in_features=64, out_features=kwargs["input_shape"]
        )

    def forward(self, features, adj=False):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code).view(-1, 11, 11)
        if adj:
            matrix = code.clone()
            return matrix
        with torch.no_grad():
            code = code * features.unsqueeze(2)
            code_sum = code.clone().sum(0)
            code_sum = (code_sum - code_sum.min()) / (code_sum.max() - code_sum.min())
            acycle = torch.matrix_power(code_sum, 11)
            acycle_loss = torch.diagonal(acycle).sum() / (len(features) * 5)
        code = code.view(-1, 121)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, acycle_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=11).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

class Sachs_Dataset(Dataset):
    def __init__(self):
        graph = pd.read_csv("sachs_dataset/adjmat.csv", index_col='source')
        df = pd.read_csv("sachs_dataset/sachs.data.txt", sep="\t")
        self.X = torch.tensor(df.values, dtype=torch.float32)
        self.X = (self.X - self.X.min(0)[0]) / (self.X.max(0)[0] - self.X.min(0)[0])
        self.graph_df = graph
        self.graph = torch.tensor(graph.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]


train_dataset = Sachs_Dataset()

# test_dataset = torchvision.datasets.MNIST(
#     root="~/torch_datasets", train=False, transform=transform, download=True
# )

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True
)

# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=32, shuffle=False, num_workers=4
# )

losses = []
a_losses = []
epochs = 100
for epoch in range(epochs):
    loss = 0
    a_loss = 0
    for batch_features in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs, acycle_loss = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        a_loss += acycle_loss
        train_loss += acycle_loss
        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)
    losses.append(loss)
    a_losses.append(a_loss.cpu())
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

plt.plot(losses, label="Normal")
plt.plot(a_losses, label="acycle")
plt.legend()
plt.show()

output_sum = torch.zeros((11, 11))
with torch.no_grad():
    for batch_features in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device

        batch_features = batch_features.to(device)

        # compute reconstructions
        output_sum = model(batch_features, adj=True).sum(0)
        # outputs2 = outputs.clone().view(-1, 11, 11)
    print("")


##
print("")