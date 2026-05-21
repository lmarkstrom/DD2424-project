import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from torchvision import transforms
from torchvision.ops import StochasticDepth

from dataHelper import loadData
from plotHelper import plotPerformance


class ConvNeXtBlock(nn.Module):
    """
    Modular ConvNeXt Block
    """

    def __init__(self, dim, CN_params, dropout_rate=0.0):
        super().__init__()
        multiplier = 4  # Expansion factor for the pointwise convolution
        # Depthwise convolution
        self.dwconv = nn.Conv2d(
            in_channels=dim, 
            out_channels=dim, 
            kernel_size=CN_params["l_conv_depthwise"]["f"], 
            padding=3, 
            groups=dim)
        self.norm = nn.LayerNorm(dim)
        # Pointwise expansion
        self.pwconv1 = nn.Conv2d(
            in_channels=dim, 
            out_channels=multiplier * dim, 
            kernel_size=CN_params["l_conv_pointwise"]["f"])
        self.gelu = nn.GELU()
        # Pointwise projection
        self.pwconv2 = nn.Conv2d(
            in_channels=multiplier * dim, 
            out_channels=dim, 
            kernel_size=CN_params["l_conv_pointwise"]["f"])
        # Stochastic depth
        self.stochastic_depth = StochasticDepth(dropout_rate, mode="row")

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW to NHWC for LayerNorm
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW for convolutions
        x = self.pwconv1(x)
        x = self.gelu(x)
        x = self.pwconv2(x)
        x = self.stochastic_depth(x)
        return residual + x


class Network(nn.Module):
    def __init__(self, LR_params, GD_params, CN_params, RE_params, train_size=49000, val_size=1000):
        """
        LR_params: Learning rate parameters
        GD_params: Gradient decent parameters
        CN_params: Convolution layer parameters
        RE_params: Regularization parameters
        train_size (int): Size of the training set
        val_size (int): Size of the validation set set
        """
        super(Network, self).__init__()
        self.CN_params = CN_params
        self.GD_params = GD_params
        self.LR_params = LR_params
        self.RE_params = RE_params
        self.downsample_layers = nn.ModuleList()  # List of patchify and spatial downsampling layers

        # ========================
        # Patchify layer
        patchify = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=CN_params["l_patchify"]["n_f"],
                kernel_size=CN_params["l_patchify"]["f"],
                stride=CN_params["l_patchify"]["s"],
            ),
            nn.GroupNorm(1, CN_params["l_patchify"]["n_f"]),  # Alternative to LayerNorm + Permute
        )
        self.downsample_layers.append(patchify)

        # Between stages downsampling (2x2 stride 2)
        for i in range(2):
            downsample_layer = nn.Sequential(
                nn.GroupNorm(1, CN_params["dims"][i]),
                nn.Conv2d(CN_params["dims"][i], CN_params["dims"][i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # ========================
        # ConvNeXt Stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(CN_params["depths"][i]):
                stage_blocks.append(ConvNeXtBlock(CN_params["dims"][i], CN_params, dropout_rate=RE_params["dropout_rates"][cur]))
                cur += 1
            self.stages.append(nn.Sequential(*stage_blocks))

        # ========================
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head_norm = nn.LayerNorm(CN_params["dims"][-1]) # Note: In ConvNeXt, LayerNorm is often applied after GAP before the linear layer

        # ========================
        # Linear layers
        self.fc = nn.Linear(CN_params["dims"][-1], CN_params["l_fc"]["out"])

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

                    # transforms.RandomAffine(degrees=0, translate=(max_shift, max_shift)),
                    
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        # ========================

        # ========================
        # Loaders
        (self.trainset_augmented, self.trainset_original, self.testloader, self.classes,) = loadData(batch_size=GD_params["n_batch"], train_transform=self.augementation)
        total_size = len(self.trainset_augmented)
        indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42)).tolist()

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]

        # Use clean and augmented datasets
        train_dataset = Subset(self.trainset_augmented, train_indices)
        val_dataset = Subset(self.trainset_original, val_indices)

        # DataLoaders
        self.trainloader = DataLoader(train_dataset, batch_size=self.GD_params["n_batch"], shuffle=True)
        self.valloader = DataLoader(val_dataset, batch_size=self.GD_params["n_batch"], shuffle=False)
        # ========================

    def forward(self, x):
        # ConvNeXt Stages: [1, 1, 3] blocks with [64, 128, 256] channels
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # Global average pooling
        x = self.gap(x)

        # Flattening (to connect to fc layer)
        x = x.view(x.size(0), -1)

        # LayerNorm before the final fc layer
        x = self.head_norm(x)

        # fc layer
        x = self.fc(x)

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
            results = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "steps": []}

        if self.LR_params["scheduler"]:
            if self.LR_params["scheduler_type"] == "step":
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                           step_size=30, 
                                                           gamma=0.1)
            elif self.LR_params["scheduler_type"] == "cosine":
                self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                               max_lr=0.01,
                                                               steps_per_epoch=len(self.trainloader),
                                                               anneal_strategy="cos",
                                                               epochs=n_epochs)
            else:
                raise ValueError(f"Unsupported scheduler type: {self.LR_params["scheduler_type"]}")

        for epoch in range(n_epochs):
            self.train()

            running_loss = 0.0
            batch_count = 0
            running_correct = 0
            running_total = 0

            for (inputs, labels) in self.trainloader:
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

                if self.LR_params["scheduler"] and self.LR_params["scheduler_type"] == "cosine":
                    self.scheduler.step()

            if self.LR_params["scheduler"] and self.LR_params["scheduler_type"] == "step":
                self.scheduler.step()

            avg_loss = running_loss / batch_count
            train_accuracy = running_correct / running_total

            if debug or plot:
                val_accuracy, val_loss = self.evaluate(self.valloader)

            if debug:
                loss_delta_new = val_loss - avg_loss
                if loss_delta_new > loss_delta and val_loss_prev < val_loss:
                    print(f"Maybe starting to overfit?: Train loss: {avg_loss:.4f} Val loss: {val_loss:.4f}, diff: {loss_delta_new:.4f}")
                loss_delta = loss_delta_new
                val_loss_prev = val_loss

                print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")
            if plot:
                results["loss"].append(avg_loss)
                results["val_loss"].append(val_loss)
                results["acc"].append(train_accuracy)
                results["val_acc"].append(val_accuracy)
                results["steps"].append(steps)
                steps += 1

        if plot:
            plotPerformance(results)