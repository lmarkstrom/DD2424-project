import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchvision import transforms

from dataHelper import loadData
from plotHelper import plotPerformance

class Network(nn.Module):
    def __init__(self, LR_params, GD_params, CN_params, RE_params, train_size=49000, val_size=1000):
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
        

        # ========================
        # Patchify layer
        self.patchify = nn.Conv2d(
            in_channels=3,
            out_channels=CN_params['l_patchify']['n_f'],
            kernel_size=CN_params['l_patchify']['f'],
            stride=CN_params['l_patchify']['s'],
        )
        # ========================
        
        # ========================
        # VGG Block-1
        self.conv1 = nn.Conv2d(
            in_channels=CN_params['l_vgg1']['n_f'], 
            out_channels=CN_params['l_vgg1']['n_f'], 
            kernel_size=CN_params['l_vgg1']['f'], 
            stride=CN_params['l_vgg1']['s'], 
            padding='same')
        self.conv2 = nn.Conv2d(
            in_channels=CN_params['l_vgg1']['n_f'], 
            out_channels=CN_params['l_vgg1']['n_f'], 
            kernel_size=CN_params['l_vgg1']['f'], 
            stride=CN_params['l_vgg1']['s'], 
            padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # ========================

        # ========================
        # VGG Block-2
        self.conv3 = nn.Conv2d(
            in_channels=CN_params['l_vgg1']['n_f'], 
            out_channels=CN_params['l_vgg2']['n_f'], 
            kernel_size=CN_params['l_vgg2']['f'], 
            stride=CN_params['l_vgg2']['s'], 
            padding='same')
        self.conv4 = nn.Conv2d(
            in_channels=(CN_params['l_vgg2']['n_f']), 
            out_channels=CN_params['l_vgg2']['n_f'], 
            kernel_size=CN_params['l_vgg2']['f'], 
            stride=CN_params['l_vgg2']['s'], 
            padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # ========================

        # ========================
        # VGG Block-3
        self.conv5 = nn.Conv2d(
            in_channels=CN_params['l_vgg2']['n_f'], 
            out_channels=CN_params['l_vgg3']['n_f'], 
            kernel_size=CN_params['l_vgg3']['f'], 
            stride=CN_params['l_vgg3']['s'], 
            padding='same')
        self.conv6 = nn.Conv2d(
            in_channels=CN_params['l_vgg3']['n_f'], 
            out_channels=CN_params['l_vgg3']['n_f'], 
            kernel_size=CN_params['l_vgg3']['f'], 
            stride=CN_params['l_vgg3']['s'], 
            padding='same')
        # ========================

        # =======================
        # Dropout
        self.dropout = nn.Dropout(RE_params["dropout_rate"])
        # ========================
        # Linear layers
        self.fc1 = nn.Linear(in_features=CN_params['l_fc1']['in'], out_features=CN_params['l_fc1']['out'])
        self.fc2 = nn.Linear(CN_params['l_fc2']['in'], CN_params['l_fc2']['out'])
         # ========================

        # ========================
        # Optimizers and metrics criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.parameters(), 
            lr=LR_params['eta'],
            weight_decay=GD_params['lam'],
        )
        # ========================
        
        # ========================
        # Augementations
        self.augementation = None
        if RE_params['augementation']:
            max_shift = RE_params['shift_max'] / GD_params['img_size']
            self.augementation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=self.RE_params['flip_prob']),
                
                transforms.RandomAffine(degrees=0, translate=(max_shift, max_shift)),
                
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        # ========================
        
        # ========================
        # Loaders
        self.trainloader, self.testloader, self.classes = loadData(batch_size=GD_params['n_batch'], augmentations=self.augementation)
        train_dataset, val_dataset = random_split(
            self.trainloader.dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        self.trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.valloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        # ========================

    def forward(self, x):
        x = self.patchify(x)
        x = F.relu(x)
        
        # ========================
        # VGG Block-1
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        # Apply maxpooling layer
        x = self.pool1(x)

        x = self.dropout(x)
        # ========================
        # ========================
        # VGG Block-2
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        
        # Apply maxpooling layer
        x = self.pool2(x)
        x = self.dropout(x)
        # ========================
        # ========================
        # VGG Block-3
        x = self.conv5(x)
        x = F.relu(x)
        
        x = self.conv6(x)
        x = F.relu(x)
        # Removed pooling in last layer as per instructions
        # ========================

        x = self.dropout(x)
        # Flattening (to connect to fc1 layer)
        x = x.view(x.size(0), -1)

        # Run through fc1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
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
        self.train()
        n_epochs = self.GD_params['n_epochs']

        loss_delta = np.inf
        val_loss_prev = np.inf
        
        if plot:
            steps = 0
            results = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [], 'steps': []}
        
        for epoch in range(n_epochs):
            running_loss = 0.0
            batch_count = 0
            
            for _i, (inputs, labels) in enumerate(self.trainloader, 0):
                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                
                running_loss += loss.item()
                batch_count += 1

            if debug:
                avg_loss = running_loss / batch_count
                val_accuracy, val_loss = self.evaluate(self.valloader)
                loss_delta_new = val_loss - avg_loss
                if loss_delta_new > loss_delta and val_loss_prev < val_loss:
                    print(f"Maybe starting to overfit?: Train loss: {avg_loss:.4f} Val loss: {val_loss:.4f}, diff: {loss_delta_new:.4f}")
                loss_delta = loss_delta_new
                val_loss_prev = val_loss

                print(f'Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}')
            if plot:
                results['loss'].append(avg_loss)
                results['val_loss'].append(val_loss)
                results['acc'].append(self.evaluate(self.trainloader)[0])
                results['val_acc'].append(val_accuracy)
                results['steps'].append(steps)
                steps += 1
            
        if plot:
            plotPerformance(results) 
