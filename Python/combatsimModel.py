import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pyperclip
import time

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
                result1, result2, result3 = int(row[24]), int(row[25]), int(row[26])  # Last 3 columns are results
                overall_result = int(row[27])  # Overall result
                self.data.append((p1_cards, p1_colors, p2_cards, p2_colors, [result1, result2, result3], overall_result))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p1_cards, p1_colors, p2_cards, p2_colors, results, overall_result = self.data[idx]
        return (torch.tensor(p1_cards).float(), torch.tensor(p1_colors).float(),
                torch.tensor(p2_cards).float(), torch.tensor(p2_colors).float(),
                torch.tensor(results, dtype=torch.long), torch.tensor(overall_result, dtype=torch.long))


class CardComparisonModel(nn.Module):  # A submodule that compares the cards of two players and predicts the outcome for one card comparison.
    def __init__(self):
        super(CardComparisonModel, self).__init__()
        layer1_features = 8
        layer2_features = 3
        self.fc1 = nn.Linear(8, layer1_features)
        self.fc2 = nn.Linear(layer1_features, layer2_features)
        self.fc3 = nn.Linear(layer2_features, 3)

    def forward(self, p1_card, p1_color, p2_card, p2_color):
        x = torch.cat((p1_card, p1_color, p2_card, p2_color), dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = (self.fc3(x))
        return x


class CardCombatModel(nn.Module):
    def __init__(self):
        super(CardCombatModel, self).__init__()
        self.model1 = CardComparisonModel()
        self.model2 = CardComparisonModel()
        self.model3 = CardComparisonModel()
        self.fc_overall1 = nn.Linear(9, 9)
        self.fc_overall2 = nn.Linear(9, 3)

    def forward(self, p1_cards, p1_colors, p2_cards, p2_colors):
        out1 = self.model1(p1_cards[:, 0].unsqueeze(1), p1_colors[:, :3], p2_cards[:, 0].unsqueeze(1), p2_colors[:, :3])
        out2 = self.model2(p1_cards[:, 1].unsqueeze(1), p1_colors[:, 3:6], p2_cards[:, 1].unsqueeze(1), p2_colors[:, 3:6])
        out3 = self.model3(p1_cards[:, 2].unsqueeze(1), p1_colors[:, 6:9], p2_cards[:, 2].unsqueeze(1), p2_colors[:, 6:9])
        combined_output = torch.cat((out1, out2, out3), dim=1)  # Concatenate the outputs
        overall_output = torch.tanh(self.fc_overall1(combined_output))
        overall_output = (self.fc_overall2(overall_output))

        return out1, out2, out3, overall_output  # Return individual and overall outcomes


def train_model(csv_file, model_save_path='combatsimModel.pth'):
    # Hyperparameters
    batch_size = 100
    learning_rate = 0.003
    epochs = 100

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

    start_time = time.time()
    print(f"Starting the training at: {start_time}")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for p1_cards, p1_colors, p2_cards, p2_colors, targets, overall_targets in train_loader:
            p1_cards, p1_colors, p2_cards, p2_colors, targets, overall_targets = \
                p1_cards.to(device), p1_colors.to(device), p2_cards.to(device), p2_colors.to(device), \
                targets.to(device), overall_targets.to(device)
            optimizer.zero_grad()
            outputs1, outputs2, outputs3, overall_output = model(p1_cards, p1_colors, p2_cards, p2_colors)

            loss1 = criterion(outputs1, targets[:, 0])
            loss2 = criterion(outputs2, targets[:, 1])
            loss3 = criterion(outputs3, targets[:, 2])
            overall_loss = criterion(overall_output, overall_targets)
            loss = loss1 + loss2 + loss3 + overall_loss

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            _, predicted1 = torch.max(outputs1.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)
            _, predicted3 = torch.max(outputs3.data, 1)
            _, predicted_overall = torch.max(overall_output.data, 1)

            total_train += targets.size(0) * 4  # Each game has 4 predictions: 3 card comparisons + 1 overall result
            correct_train += (predicted1 == targets[:, 0]).sum().item()
            correct_train += (predicted2 == targets[:, 1]).sum().item()
            correct_train += (predicted3 == targets[:, 2]).sum().item()
            correct_train += (predicted_overall == overall_targets).sum().item()

        train_losses.append(epoch_train_loss / len(train_loader))
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            epoch_test_loss = 0.0
            correct_test = 0
            total_test = 0

            for p1_cards, p1_colors, p2_cards, p2_colors, targets, overall_targets in test_loader:
                p1_cards, p1_colors, p2_cards, p2_colors, targets, overall_targets = \
                    p1_cards.to(device), p1_colors.to(device), p2_cards.to(device), p2_colors.to(device), \
                    targets.to(device), overall_targets.to(device)
                outputs1, outputs2, outputs3, overall_output = model(p1_cards, p1_colors, p2_cards, p2_colors)

                loss1 = criterion(outputs1, targets[:, 0])
                loss2 = criterion(outputs2, targets[:, 1])
                loss3 = criterion(outputs3, targets[:, 2])
                overall_loss = criterion(overall_output, overall_targets)
                loss = loss1 + loss2 + loss3 + overall_loss
                epoch_test_loss += loss.item()

                _, predicted1 = torch.max(outputs1.data, 1)
                _, predicted2 = torch.max(outputs2.data, 1)
                _, predicted3 = torch.max(outputs3.data, 1)
                _, predicted_overall = torch.max(overall_output.data, 1)

                total_test += targets.size(0) * 4  # Each game has 4 predictions: 3 card comparisons + 1 overall result
                correct_test += (predicted1 == targets[:, 0]).sum().item()
                correct_test += (predicted2 == targets[:, 1]).sum().item()
                correct_test += (predicted3 == targets[:, 2]).sum().item()
                correct_test += (predicted_overall == overall_targets).sum().item()

            test_losses.append(epoch_test_loss / len(test_loader))
            test_accuracy = 100 * correct_test / total_test
            test_accuracies.append(test_accuracy)

        if epoch == 0:
            lastepochs_train_losses = []
            lastepochs_test_losses = []

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracy:.2f}%")
        if epochs - epoch <= 4:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracy:.2f}%")

            lastepochs_train_losses.append(train_losses[-1])
            lastepochs_test_losses.append(test_losses[-1])
        if epochs - epoch == 1:
            end_time = time.time()
            print(f"Ending the training at {end_time}")
            print(f"Average training loss: {(sum(train_losses)/len(train_losses)):4f} | Average training loss of last 4 epochs: {(sum(lastepochs_train_losses)/len(lastepochs_train_losses)):4f} | Lowest training loss: {min(test_losses):4f}")
            print(f"Average test loss: {(sum(test_losses)/len(test_losses)):4f} | Average test loss of last 4 epochs: {(sum(lastepochs_test_losses)/len(lastepochs_test_losses)):4f} | Lowest test loss: {min(test_losses):4f}")

            elapsed_time = end_time - start_time
            print(f"Time elapsed: {elapsed_time}")

            pyperclip.copy(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, "
                           f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracy:.2f}% | Time elapsed: {elapsed_time:.2f} sec")
            print("Last epoch stats and time elapsed copied to clipboard")

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
    p1_colors = [int(input("Enter p1 color 1 (1 for green, 2 for red, 3 for blue): ")),
                 int(input("Enter p1 color 2 (1 for green, 2 for red, 3 for blue): ")),
                 int(input("Enter p1 color 3 (1 for green, 2 for red, 3 for blue): "))]
    p2_cards = [int(input("Enter p2 card 1: ")), int(input("Enter p2 card 2: ")), int(input("Enter p2 card 3: "))]
    p2_colors = [int(input("Enter p2 color 1 (1 for green, 2 for red, 3 for blue): ")),
                 int(input("Enter p2 color 2 (1 for green, 2 for red, 3 for blue): ")),
                 int(input("Enter p2 color 3 (1 for green, 2 for red, 3 for blue): "))]
    # Convert colors to one-hot encoding
    p1_colors_onehot = [0] * 9
    p2_colors_onehot = [0] * 9
    for i in range(3):
        p1_colors_onehot[i * 3 + (p1_colors[i] - 1)] = 1
        p2_colors_onehot[i * 3 + (p2_colors[i] - 1)] = 1
    return (p1_cards, p1_colors_onehot, p2_cards, p2_colors_onehot)


def predict(model, p1_cards, p1_colors, p2_cards, p2_colors):
    model.eval()
    with torch.no_grad():
        # Convert lists to tensors and reshape them for the model
        p1_cards_tensor = torch.tensor(p1_cards).unsqueeze(0).float().to(device)
        p1_colors_tensor = torch.tensor(p1_colors).unsqueeze(0).float().to(device)
        p2_cards_tensor = torch.tensor(p2_cards).unsqueeze(0).float().to(device)
        p2_colors_tensor = torch.tensor(p2_colors).unsqueeze(0).float().to(device)

        # Ensure the colors are reshaped correctly to match the model's expectation
        p1_colors_tensor = p1_colors_tensor.view(1, -1)
        p2_colors_tensor = p2_colors_tensor.view(1, -1)

        outputs1, outputs2, outputs3, overall_output = model(p1_cards_tensor, p1_colors_tensor, p2_cards_tensor, p2_colors_tensor)
        _, predicted1 = torch.max(outputs1.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        _, predicted3 = torch.max(outputs3.data, 1)
        _, predicted_overall = torch.max(overall_output.data, 1)

        results = [predicted1.item(), predicted2.item(), predicted3.item()]

        overall_result = predicted_overall.item()  # Use the overall prediction directly

        return overall_result, results

def manual_input(model):
    p1_cards, p1_colors, p2_cards, p2_colors = get_manual_input()
    overall_result, results = predict(model, p1_cards, p1_colors, p2_cards, p2_colors)

    print(f"Card comparison results: {results}")
    if overall_result == 0:
        print("Prediction: Player 1 wins!")
    elif overall_result == 1:
        print("Prediction: Player 2 wins!")
    else:
        print("Prediction: It's a draw!")



if __name__ == "__main__":
    answer = input("Would you like to retrain the model? (y/n) ")
    if answer == "y":
        train_model('results.csv')
        answer = input("Would you like to manually test the model? (y/n) ")
        if answer == "y":
            while True:
                manual_input(load_model())
    if answer == "n":
        answer = input("Would you like to manually test the model? (y/n) ")
        if answer == "y":
            while True:
                manual_input(load_model())

