import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataHelper import loadData
from layers.ecaBlock import ECABlock
from layers.seBlock import SEBlock
from plotHelper import plotPerformance


class Network(nn.Module):
    def __init__(
        self,
        LR_params,
        GD_params,
        CN_params,
        RE_params,
        train_size=49000,
        val_size=1000,
    ):
        """
        LR_params: Learning rate parameters
        GD_params: Gradient decent parameters
        CN_params: Convolution layer parameters
        RE_params: Regularization parameters
        train_size (int): Size of the training set
        val_size (int): Size of the validation set set

        Initializes the network layers layers and fully connected layers
        for image handling, hardcoded for (32 // f) ** 2 atm atm
        """
        super(Network, self).__init__()
        self.CN_params = CN_params
        self.GD_params = GD_params
        self.LR_params = LR_params
        self.RE_params = RE_params

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        # ========================
        # Patchify layer
        self.patchify = nn.Conv2d(
            in_channels=3,
            out_channels=CN_params["l_patchify"]["n_f"],
            kernel_size=CN_params["l_patchify"]["f"],
            stride=CN_params["l_patchify"]["s"],
        )
        self.bn_patch = nn.BatchNorm2d(CN_params["l_patchify"]["n_f"])
        # ========================

        # ========================
        # VGG Block-1
        self.conv1 = nn.Conv2d(
            in_channels=CN_params["l_vgg1"]["n_f"],
            out_channels=CN_params["l_vgg1"]["n_f"],
            kernel_size=CN_params["l_vgg1"]["f"],
            stride=CN_params["l_vgg1"]["s"],
            padding="same",
        )
        self.bn1 = nn.BatchNorm2d(CN_params["l_vgg1"]["n_f"])

        self.conv2 = nn.Conv2d(
            in_channels=CN_params["l_vgg1"]["n_f"],
            out_channels=CN_params["l_vgg1"]["n_f"],
            kernel_size=CN_params["l_vgg1"]["f"],
            stride=CN_params["l_vgg1"]["s"],
            padding="same",
        )
        self.bn2 = nn.BatchNorm2d(CN_params["l_vgg1"]["n_f"])

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # ========================

        # ========================
        # VGG Block-2
        self.conv3 = nn.Conv2d(
            in_channels=CN_params["l_vgg1"]["n_f"],
            out_channels=CN_params["l_vgg2"]["n_f"],
            kernel_size=CN_params["l_vgg2"]["f"],
            stride=CN_params["l_vgg2"]["s"],
            padding="same",
        )
        self.bn3 = nn.BatchNorm2d(CN_params["l_vgg2"]["n_f"])

        self.conv4 = nn.Conv2d(
            in_channels=(CN_params["l_vgg2"]["n_f"]),
            out_channels=CN_params["l_vgg2"]["n_f"],
            kernel_size=CN_params["l_vgg2"]["f"],
            stride=CN_params["l_vgg2"]["s"],
            padding="same",
        )
        self.bn4 = nn.BatchNorm2d(CN_params["l_vgg2"]["n_f"])

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # ========================

        # ========================
        # VGG Block-3
        self.conv5 = nn.Conv2d(
            in_channels=CN_params["l_vgg2"]["n_f"],
            out_channels=CN_params["l_vgg3"]["n_f"],
            kernel_size=CN_params["l_vgg3"]["f"],
            stride=CN_params["l_vgg3"]["s"],
            padding="same",
        )
        self.bn5 = nn.BatchNorm2d(CN_params["l_vgg3"]["n_f"])

        self.conv6 = nn.Conv2d(
            in_channels=CN_params["l_vgg3"]["n_f"],
            out_channels=CN_params["l_vgg3"]["n_f"],
            kernel_size=CN_params["l_vgg3"]["f"],
            stride=CN_params["l_vgg3"]["s"],
            padding="same",
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bn6 = nn.BatchNorm2d(CN_params["l_vgg3"]["n_f"])
        # ========================

        # ========================
        # Linear layers
        self.fc2 = nn.Linear(CN_params["l_fc2"]["in"], CN_params["l_fc2"]["out"])
        # ========================

        # ========================
        # SE-Layers
        self.use_se = RE_params["se"]

        self.se1 = SEBlock(
            channels=CN_params["l_vgg1"]["n_f"], reduction=CN_params["l_vgg1"]["r"]
        )
        self.se2 = SEBlock(
            channels=CN_params["l_vgg2"]["n_f"], reduction=CN_params["l_vgg2"]["r"]
        )
        self.se3 = SEBlock(
            channels=CN_params["l_vgg3"]["n_f"], reduction=CN_params["l_vgg3"]["r"]
        )

        # ECA-Layers
        self.use_eca = RE_params["eca"]
        self.eca1 = ECABlock(channels=CN_params["l_vgg1"]["n_f"])
        self.eca2 = ECABlock(channels=CN_params["l_vgg2"]["n_f"])
        self.eca3 = ECABlock(channels=CN_params["l_vgg3"]["n_f"])

        # =======================
        # Dropout
        self.dropout = RE_params["dropout"]
        self.dropout1 = nn.Dropout(RE_params["dropout_rates"][0])
        self.dropout2 = nn.Dropout(RE_params["dropout_rates"][1])
        self.dropout3 = nn.Dropout(RE_params["dropout_rates"][2])
        self.dropout4 = nn.Dropout(RE_params["dropout_rates"][3])
        # ========================

        # ========================
        # Optimizers and metrics criterion
        if self.RE_params["label_smoothing"]:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.RE_params["smoothing_factor"]
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=LR_params["eta"],
            weight_decay=GD_params["lam"],
        )
        # ========================

        # ========================
        # Augementations
        self.augementation = None
        if RE_params["augementation"]:
            self.augementation = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=self.RE_params["flip_prob"]),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomRotation(self.RE_params["rotation"]),
                    # transforms.RandomAffine(degrees=0, translate=(max_shift, max_shift)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        # ========================

        # ========================
        # Loaders
        (
            self.trainset_augmented,
            self.trainset_original,
            self.testloader,
            self.classes,
        ) = loadData(
            batch_size=GD_params["n_batch"], train_transform=self.augementation
        )
        total_size = len(self.trainset_augmented)
        indices = torch.randperm(
            total_size, generator=torch.Generator().manual_seed(42)
        ).tolist()

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]

        # Use clean and augmented datasets
        train_dataset = Subset(self.trainset_augmented, train_indices)
        val_dataset = Subset(self.trainset_original, val_indices)

        # DataLoaders
        self.trainloader = DataLoader(
            train_dataset,
            batch_size=self.GD_params["n_batch"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        self.valloader = DataLoader(
            val_dataset,
            batch_size=self.GD_params["n_batch"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        # ========================
        self.to(self.device)

    def forward(self, x):
        x = self.patchify(x)
        x = self.bn_patch(x)
        x = F.relu(x)

        # ========================
        # VGG Block-1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        if self.use_se:
            x = self.se1(x)
        if self.use_eca:
            x = self.eca1(x)

        x = self.pool1(x)
        if self.dropout:
            x = self.dropout1(x)
        # ========================
        # ========================
        # VGG Block-2
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        if self.use_se:
            x = self.se2(x)
        if self.use_eca:
            x = self.eca2(x)

        # Apply maxpooling layer
        x = self.pool2(x)
        if self.dropout:
            x = self.dropout2(x)
        # ========================
        # ========================
        # VGG Block-3
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        if self.use_se:
            x = self.se3(x)
        if self.use_eca:
            x = self.eca3(x)

        if self.dropout:
            x = self.dropout3(x)
        # Removed pooling in last layer as per instructions
        # ========================

        x = self.gap(x)

        # Flattening (to connect to fc layer)
        x = x.view(x.size(0), -1)

        if self.dropout:
            x = self.dropout4(x)
        # fc2 layer
        x = self.fc2(x)

        return x

    def evaluate(self, dataset):
        """
        Evaluate the trained network on the trained parameters

        self: The network itself
        dataset: The inserted dataset

        Returns: Accuracy and loss for the model
        """
        self.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            count = 0
            for inputs, labels in dataset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                count += 1
        accuracy = correct / total
        loss /= count
        self.train()
        return accuracy, loss

    def trainModel(self, debug=True, plot=False):
        """
        Train model using CLR with increasing cycle lengths
        """

        n_epochs = self.GD_params["n_epochs"]
        loss_delta = np.inf
        val_loss_prev = np.inf

        if plot:
            steps = 0
            results = {
                "loss": [],
                "val_loss": [],
                "acc": [],
                "val_acc": [],
                "steps": [],
            }

        if self.LR_params["scheduler"]:
            if self.LR_params["scheduler_type"] == "step":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=30, gamma=0.1
                )
            elif self.LR_params["scheduler_type"] == "cosine":
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.LR_params["max_lr"],
                    steps_per_epoch=len(self.trainloader),
                    anneal_strategy="cos",
                    epochs=n_epochs,
                )
            else:
                raise ValueError(
                    f"Unsupported scheduler type: {self.LR_params['scheduler_type']}"
                )

        for epoch in range(n_epochs):
            self.train()

            running_loss = 0.0
            batch_count = 0
            running_correct = 0
            running_total = 0

            for inputs, labels in self.trainloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_total += labels.size(0)
                running_correct += (predicted == labels).sum().item()
                batch_count += 1

                if (
                    self.LR_params["scheduler"]
                    and self.LR_params["scheduler_type"] == "cosine"
                ):
                    self.scheduler.step()

            if (
                self.LR_params["scheduler"]
                and self.LR_params["scheduler_type"] == "step"
            ):
                self.scheduler.step()

            avg_loss = running_loss / batch_count
            train_accuracy = running_correct / running_total

            if debug or plot:
                val_accuracy, val_loss = self.evaluate(self.valloader)

            if debug:
                if (train_accuracy - val_accuracy) > 0.05 and val_loss > val_loss_prev:
                    print(
                        f"(!) Possibly overfitting : Train Acc {train_accuracy:.2f} vs Val Acc {val_accuracy:.2f}"
                    )

                    val_loss_prev = val_loss

                print(
                    f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}"
                )
            if plot:
                results["loss"].append(avg_loss)
                results["val_loss"].append(val_loss)
                results["acc"].append(train_accuracy)
                results["val_acc"].append(val_accuracy)
                results["steps"].append(steps)
                steps += 1

        if plot:
            plotPerformance(results)
