import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.classifier.convnet import ConvNeXt
import inspect

ITOS = {0: "backhand", 1: "forehand", 2: "serve", 3: "undefine"}
STOI = {"backhand": 0, "forehand": 1, "serve": 2, "undefine": 3}


class DataLoader:
    def __init__(self, img_folder, labels):
        self.img_folder = img_folder
        files = sorted(
            [f for f in os.listdir(self.img_folder) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        self.files = np.array(files)

        self.target = np.loadtxt(labels, delimiter=",", dtype=str)
        assert len(self.files) == len(self.target)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.files,
            self.target,
            test_size=0.1,
            stratify=self.target,
            shuffle=True,
            random_state=42,
        )

        self.img_process_train = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.RandomInvert(p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4093, 0.4325, 0.4644], std=[0.1452, 0.1499, 0.1524]
                ),
            ]
        )

        self.img_process_val = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4093, 0.4325, 0.4644], std=[0.1452, 0.1499, 0.1524]
                ),
            ]
        )

        self.B = 64
        self.current_batch_idx_train = 0
        self.current_batch_idx_val = 0
        self.num_of_epochs = 0

    def _transform(self, clip_files, mode):
        imgs = []
        for f in clip_files:
            img = Image.open(os.path.join(self.img_folder, f)).convert("RGB")
            frame = (
                self.img_process_train(img)
                if mode == "train"
                else self.img_process_val(img)
            )
            imgs.append(frame)

        return torch.stack(imgs, dim=0)  # (B, 3, 128, 128)

    def get_next_batch(self):
        img_files = self.X_train[
            self.current_batch_idx_train : self.current_batch_idx_train + self.B
        ]
        labels = self.y_train[
            self.current_batch_idx_train : self.current_batch_idx_train + self.B
        ]

        y = torch.tensor([STOI[s] for s in labels])  # (B,)
        x = self._transform(img_files, mode="train")  # (B, 3, 128, 128)

        self.current_batch_idx_train += self.B
        if self.current_batch_idx_train >= self.X_train.shape[0]:
            # reset and reshuffle after an epoch
            self.num_of_epochs += 1
            self.current_batch_idx_train = 0
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train)

        return x, y  # (B, 3, 128, 128), (B,)

    def get_next_batch_val(self):
        img_files = self.X_val[
            self.current_batch_idx_val : self.current_batch_idx_val + self.B
        ]
        labels = self.y_val[
            self.current_batch_idx_val : self.current_batch_idx_val + self.B
        ]

        y = torch.tensor([STOI[s] for s in labels])  # (B,)
        x = self._transform(img_files, mode="val")  # (B, 3, 128, 128)

        end_of_val = False
        self.current_batch_idx_val += self.B
        if self.current_batch_idx_val >= self.X_val.shape[0]:
            end_of_val = True
            self.current_batch_idx_val = 0

        return x, y, end_of_val


def train(img_folder, target_folder, num_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_epochs = num_epochs
    loader = DataLoader(img_folder, target_folder)

    model = ConvNeXt()
    model.to(device)
    raw_model = model
    model = torch.compile(model)

    learning_rate = 4e-4

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda"
    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999), fused=use_fused
    )

    model.train()

    loss_plot = []
    val_acc_plot = []
    best_acc = 0.0

    while loader.num_of_epochs < total_epochs:

        current_epoch = loader.num_of_epochs
        optimizer.zero_grad()

        x, y = loader.get_next_batch()
        x, y = x.to(device), y.to(device)

        logit = model(x)  # (N, 4)
        loss = F.cross_entropy(logit, y)

        loss.backward()
        optimizer.step()

        if loader.num_of_epochs != current_epoch:

            model.eval()
            correct = torch.zeros((), dtype=torch.long, device=device)
            num_examples = 0

            end_of_validation = False

            while not end_of_validation:
                x_val, y_val, end_of_validation = loader.get_next_batch_val()
                x_val, y_val = x_val.to(device), y_val.to(device)

                with torch.no_grad():
                    logits = model(x_val)  # (B, 4)

                pred = logits.argmax(dim=1)  # (B,)
                num_examples += len(y_val)
                correct += (pred == y_val).sum()

            acc = (correct.float() / num_examples).item()

            if acc > best_acc:
                best_acc = acc
                torch.save(raw_model.state_dict(), "model.pth")

            loss_plot.append(loss.item())
            val_acc_plot.append(acc)

            print(
                f"epoch {loader.num_of_epochs:5d} | loss: {loss.item():.6f} | val acc: {acc:.6f}"
            )

            model.train()

    np.save(loss_plot.npy, loss_plot)
    np.save(val_acc_plot.npy, val_acc_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train patch classifier")
    parser.add_argument("folder", help="image patch folder")
    parser.add_argument("target", help="target label file")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    args = parser.parse_args()
    train(args.folder, args.target, args.epochs)
