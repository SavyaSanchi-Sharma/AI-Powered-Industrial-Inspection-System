# src/aircraft/train_aircraft.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .aircraft_dataset import AircraftDamageDataset
from .aircraft_model import AircraftDamageModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AircraftDamageDataset(
    root_dir="aircraft_damage_dataset_v1/train"
)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AircraftDamageModel().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    total_loss = 0
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.float().to(DEVICE)

        preds = model(imgs).squeeze()
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "aircraft_damage_model.pth")
