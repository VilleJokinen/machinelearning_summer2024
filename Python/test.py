import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CardCombatDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.p1_cards = self.data.iloc[:, :3].values
        self.p2_cards = self.data.iloc[:, 3:6].values
        self.target = self.data.iloc[:, -1].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p1_cards = self.p1_cards[idx]
        p2_cards = self.p2_cards[idx]
        target = self.target[idx]
        p1_cards_tensor = torch.tensor(p1_cards, dtype=torch.float32)
        p2_cards_tensor = torch.tensor(p2_cards, dtype=torch.float32)
        target_tensor = torch.tensor(target - 1, dtype=torch.long)
        return p1_cards_tensor, p2_cards_tensor, target_tensor

class CardCombatModel(nn.Module):
    def __init__(self):
        super(CardCombatModel, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, p1_cards, p2_cards):
        x = torch.cat((p1_cards, p2_cards), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for p1_cards, p2_cards, target in dataloader:
            outputs = model(p1_cards, p2_cards)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

csv_file = 'results.csv'
dataset = CardCombatDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = CardCombatModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
training_loss = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for p1_cards, p2_cards, target in dataloader:
        optimizer.zero_grad()
        outputs = model(p1_cards, p2_cards)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    training_loss.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Plot the training loss
plt.plot(range(1, num_epochs + 1), training_loss, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

# Evaluate the model
accuracy = evaluate(model, dataloader)
print(f'Accuracy: {accuracy}%')
