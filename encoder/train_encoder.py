import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encoder.encoder_model import ContextEncoder

def load_data(data_dir="data/trajectories"):
    label_map = {
        "cartpole_light.pt": 0,
        "cartpole_medium.pt": 1,
        "cartpole_heavy.pt": 2
    }
    X, y = [], []
    for fname, label in label_map.items():
        path = os.path.join(data_dir, fname)
        data = torch.load(path)  # shape: (num_traj, time, features)
        X.append(data)
        y.append(torch.full((data.shape[0],), label, dtype=torch.long))
    X = torch.cat(X)
    y = torch.cat(y)
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

def train_encoder():
    model = ContextEncoder(input_dim=45, hidden_dim=128, output_dim=3)  # 3-way classifier
    dataloader = load_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = correct / total
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Accuracy = {acc:.2%}")

    torch.save(model.state_dict(), "encoder/context_encoder.pth")
    print("✅ Saved encoder to encoder/context_encoder.pth")

if __name__ == "__main__":
    train_encoder()