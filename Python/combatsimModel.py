import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pyperclip

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(1)
torch.manual_seed(1)

class Player:
    def __init__(self, cards, points):
        self.cards = cards
        self.points = points

class CardCombatDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                p1_cards = list(map(int, row[:3]))  # First 3 columns are p1 cards
                p1_colors = list(map(int, row[3:12]))  # Next 9 columns are p1 one-hot colors
                p2_cards = list(map(int, row[12:15]))  # Next 3 columns are p2 cards
                p2_colors = list(map(int, row[15:24]))  # Next 9 columns are p2 one-hot colors
                result = int(row[24])  # Last column is result
                result -= 1  # Adjust result to be in range [0, 2] instead of [1, 2, 3]
                self.data.append((p1_cards, p1_colors, p2_cards, p2_colors, result))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p1_cards, p1_colors, p2_cards, p2_colors, result = self.data[idx]
        return (torch.tensor(p1_cards), torch.tensor(p1_colors),
                torch.tensor(p2_cards), torch.tensor(p2_colors),
                torch.tensor(result, dtype=torch.long))

class CardCombatModel(nn.Module):
    def __init__(self):
        super(CardCombatModel, self).__init__()
        self.fc1 = nn.Linear(24, 32)
        self.fc2 = nn.Linear(32, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 3)

    def forward(self, p1_cards, p1_colors, p2_cards, p2_colors):
        x = torch.cat((p1_cards, p1_colors, p2_cards, p2_colors), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(csv_file, model_save_path='combatsimModel.pth'):
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.0025
    epochs = 400

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

        for p1_cards, p1_colors, p2_cards, p2_colors, targets in train_loader:
            p1_cards, p1_colors, p2_cards, p2_colors, targets = p1_cards.to(device), p1_colors.to(device), p2_cards.to(device), p2_colors.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(p1_cards.float(), p1_colors.float(), p2_cards.float(), p2_colors.float())
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

            for p1_cards, p1_colors, p2_cards, p2_colors, targets in test_loader:
                p1_cards, p1_colors, p2_cards, p2_colors, targets = p1_cards.to(device), p1_colors.to(device), p2_cards.to(device), p2_colors.to(device), targets.to(device)
                outputs = model(p1_cards.float(), p1_colors.float(), p2_cards.float(), p2_colors.float())
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
            pyperclip.copy(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracy:.2f}%")
            print("Last epoch stats copied to clipboard")

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
    p1_colors = [int(input("Enter p1 color 1: ")), int(input("Enter p1 color 2: ")), int(input("Enter p1 color 3: "))]
    p2_cards = [int(input("Enter p2 card 1: ")), int(input("Enter p2 card 2: ")), int(input("Enter p2 card 3: "))]
    p2_colors = [int(input("Enter p2 color 1: ")), int(input("Enter p2 color 2: ")), int(input("Enter p2 color 3: "))]
    return (torch.tensor(p1_cards), torch.tensor(p1_colors),
            torch.tensor(p2_cards), torch.tensor(p2_colors))

def predict(model, p1_cards, p1_colors, p2_cards, p2_colors):
    model.eval()
    with torch.no_grad():
        p1_cards_tensor = torch.tensor(p1_cards).unsqueeze(0).float()
        p1_colors_tensor = torch.tensor(p1_colors).unsqueeze(0).float()
        p2_cards_tensor = torch.tensor(p2_cards).unsqueeze(0).float()
        p2_colors_tensor = torch.tensor(p2_colors).unsqueeze(0).float()
        outputs = model(p1_cards_tensor.to(device), p1_colors_tensor.to(device),
                        p2_cards_tensor.to(device), p2_colors_tensor.to(device))
        _, predicted = torch.max(outputs.data, 1)
        result = predicted.item()
        return result

def manual_input(model):
    p1_cards, p1_colors, p2_cards, p2_colors = get_manual_input()
    result = predict(model, p1_cards, p1_colors, p2_cards, p2_colors)
    if result == 0:
        print("Prediction: Player 1 wins!")
    elif result == 1:
        print("Prediction: Player 2 wins!")
    elif result == 2:
        print("Prediction: It's a draw!")

if __name__ == "__main__":
    # Uncomment to train the model
    train_model('results.csv')
    # Uncomment to test with manual input
    # manual_input(CardCombatModel().to(device))
    pass
