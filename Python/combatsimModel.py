import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(1)
torch.manual_seed(1)

# Player class
class Player:
    def __init__(self, cards, points):
        self.cards = cards
        self.points = points

# Custom dataset class
class CardCombatDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                p1_cards = list(map(int, row[:3]))  # First 3 columns are p1 cards
                p2_cards = list(map(int, row[3:6]))  # Next 3 columns are p2 cards
                result = int(row[6])  # Last column is result
                result -= 1  # Adjust result to be in range [0, 2] instead of [1, 2, 3]
                self.data.append((p1_cards, p2_cards, result))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p1_cards, p2_cards, result = self.data[idx]
        return torch.tensor(p1_cards), torch.tensor(p2_cards), torch.tensor(result, dtype=torch.long)

# Neural network model
class CardCombatModel(nn.Module):
    def __init__(self):
        super(CardCombatModel, self).__init__()
        self.fc1 = nn.Linear(6, 12)  # Input size 6 (3 cards each for p1 and p2)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 3)  # Output 3 classes (1, 2, 3 for result)

    def forward(self, p1_cards, p2_cards):
        x = torch.cat((p1_cards, p2_cards), dim=1)  # Concatenate p1 and p2 cards
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model
def train_model(csv_file, model_save_path='combatsimModel.pth'):
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.005
    epochs = 50

    # Load dataset and split into train and test
    dataset = CardCombatDataset(csv_file)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = CardCombatModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for p1_cards, p2_cards, targets in train_loader:
            p1_cards, p2_cards, targets = p1_cards.to(device), p2_cards.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(p1_cards.float(), p2_cards.float())  # Convert to float if needed
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_losses.append(epoch_train_loss / len(train_loader))
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            epoch_test_loss = 0.0
            correct_test = 0
            total_test = 0

            for p1_cards, p2_cards, targets in test_loader:
                p1_cards, p2_cards, targets = p1_cards.to(device), p2_cards.to(device), targets.to(device)
                outputs = model(p1_cards.float(), p2_cards.float())  # Convert to float if needed
                loss = criterion(outputs, targets)
                epoch_test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

            test_losses.append(epoch_test_loss / len(test_loader))
            test_accuracy = 100 * correct_test / total_test
            test_accuracies.append(test_accuracy)

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracy:.2f}%")
        elif epoch == epochs - 1:
            print(f"Epoch [LAST], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracy:.2f}%")

        if epoch == 25:
            for g in optimizer.param_groups:
                g['lr'] = 0.001

        if epoch == 40:
            for g in optimizer.param_groups:
                g['lr'] = 0.0005

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(epochs), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def load_model(model_path='combatsimModel.pth'):
    model = CardCombatModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_manual_input():
    p1_cards = [int(input("Enter p1 card 1: ")), int(input("Enter p1 card 2: ")), int(input("Enter p1 card 3: "))]
    p2_cards = [int(input("Enter p2 card 1: ")), int(input("Enter p2 card 2: ")), int(input("Enter p2 card 3: "))]
    return torch.tensor(p1_cards), torch.tensor(p2_cards)

def predict(model, p1_cards, p2_cards):
    model.eval()
    with torch.no_grad():
        p1_cards_tensor = torch.tensor(p1_cards).unsqueeze(0).float()
        p2_cards_tensor = torch.tensor(p2_cards).unsqueeze(0).float()
        outputs = model(p1_cards_tensor.to(device), p2_cards_tensor.to(device))
        _, predicted = torch.max(outputs.data, 1)
        result = predicted.item()
        return result

def manual_input():
    p1_cards, p2_cards = get_manual_input()
    result = predict(model, p1_cards, p2_cards)
    if result == 0:
        print("Prediction: Player 1 wins!")
    elif result == 1:
        print("Prediction: Player 2 wins!")
    elif result == 2:
        print("Prediction: It's a draw!")

if __name__ == "__main__":
    # Uncomment to train the model
    # train_model('results.csv')
    pass
