import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import YOLOv3
from yolo_loss import YOLOHelmetLoss
from dataset import HelmetDataset  #

import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 20
NUM_CLASSES = 2
INPUT_SIZE = 416

def train():
    model = YOLOv3(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = YOLOHelmetLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_dataset = HelmetDataset(
        root="",
        annotations="",
        img_size=INPUT_SIZE
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_box, total_obj, total_cls = 0, 0, 0, 0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            preds = model(imgs)
            loss, loss_dict = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_box  += loss_dict["box_loss"]
            total_obj  += loss_dict["obj_loss"]
            total_cls  += loss_dict["cls_loss"]

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Total Loss: {total_loss/len(train_loader):.4f} "
              f"Box: {total_box/len(train_loader):.4f} "
              f"Obj: {total_obj/len(train_loader):.4f} "
              f"Cls: {total_cls/len(train_loader):.4f}")

        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/yolov3_epoch{epoch+1}.pth")

if __name__ == "__main__":
    train()
