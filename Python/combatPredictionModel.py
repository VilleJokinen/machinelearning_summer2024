import torch
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data import Dataset, DataLoader

class CardCombatDataset(Dataset):
    def __init__(self, csv_file):
        # Load the data
        self.data = pd.read_csv(csv_file)
        # Separate features and target
        self.p1_cards = self.data.iloc[:, :3].values  # First 3 columns for p1's cards
        self.p2_cards = self.data.iloc[:, 3:6].values  # Next 3 columns for p2's cards
        self.target = self.data.iloc[:, -1].values  # Last column for the result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the feature and target for a given index
        p1_cards = self.p1_cards[idx]
        p2_cards = self.p2_cards[idx]
        target = self.target[idx]
        # Convert to PyTorch tensors
        p1_cards_tensor = torch.tensor(p1_cards, dtype=torch.float32)
        p2_cards_tensor = torch.tensor(p2_cards, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return p1_cards_tensor, p2_cards_tensor, target_tensor

# Create the dataset and dataloader
csv_file = 'results.csv'
dataset = CardCombatDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

