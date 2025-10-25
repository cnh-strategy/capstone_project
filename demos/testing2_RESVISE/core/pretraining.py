import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================
PRETRAINING FUNCTION
# =====================================================

def pretraining(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    lr=0.001,
    batch_size=32,
    loss_fn="MSE",
    patience=20,
    save_path=None,
    log_dir="runs",
    scheduler_patience=5,
    scheduler_factor=0.5,
    min_lr=1e-6
):

    # -----------------------------------------------
    # Device & Model Setup
    # -----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Logging setup (TensorBoard)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # -----------------------------------------------
    # Custom Loss 정의
    # -----------------------------------------------
    def get_loss_fn(name):
        name = name.upper()
        if name == "MSE":
            return nn.MSELoss()
        elif name == "MAE":
            return nn.L1Loss()
        elif name == "HUBER":
            return nn.SmoothL1Loss()
        elif name == "MAPE":
            class MAPELoss(nn.Module):
                def __init__(self, eps=1e-8):
                    super().__init__()
                    self.eps = eps
                def forward(self, preds, targets):
                    loss = torch.abs((targets - preds) / (targets + self.eps))
                    return torch.mean(loss) * 100
            return MAPELoss()
        else:
            raise ValueError(f"Unsupported loss: {name}")

    criterion = get_loss_fn(loss_fn)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=min_lr, verbose=True
    )

    # -----------------------------------------------
    # Dataset / Dataloader
    # -----------------------------------------------
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------------------------
    # Training Loop
    # -----------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X).squeeze()
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        preds_all, targets_all = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X).squeeze()
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                preds_all.append(preds.cpu().numpy())
                targets_all.append(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        preds_all = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)

        # ✅ 확장된 평가 지표
        rmse = math.sqrt(mean_squared_error(targets_all, preds_all))
        mae = mean_absolute_error(targets_all, preds_all)
        mape = np.mean(np.abs((targets_all - preds_all) / (targets_all + 1e-8))) * 100
        r2 = r2_score(targets_all, preds_all)

        # 기록
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/RMSE", rmse, epoch)
        writer.add_scalar("Metrics/MAE", mae, epoch)
        writer.add_scalar("Metrics/MAPE", mape, epoch)
        writer.add_scalar("Metrics/R2", r2, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Scheduler 업데이트
        scheduler.step(val_loss)

        # ✅ Early Stopping + Checkpoint 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"💾 Best model saved at {save_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

        # 로그 출력
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | Train={train_loss:.6f} | Val={val_loss:.6f} "
                f"| RMSE={rmse:.4f} | MAE={mae:.4f} | MAPE={mape:.2f}% | R2={r2:.3f}"
            )

    writer.close()

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "metrics": {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    }
