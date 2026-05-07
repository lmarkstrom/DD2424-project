import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataHelper import loadData
from plotHelper import plotPerformance

class Network(nn.Module):
    def __init__(self, LR_params, GD_params, CN_params, train_size=49000, val_size=1000):
        """
        LR_params: Learning rate parameters
        GD_params: Gradient decent parameters
        CN_params: Convolution layer parameters

        Initializes the network layers layers and fully connected layers
        for image handling, hardcoded for (32 // f) ** 2 atm atm
        """
        super(Network, self).__init__()
        self.CN_params = CN_params
        self.GD_params = GD_params
        self.LR_params = LR_params

        self.patchify = nn.Conv2d(
            in_channels=3,
            out_channels=CN_params['l_patchify']['n_f'],
            kernel_size=CN_params['l_patchify']['f'],
            stride=CN_params['l_patchify']['s'],
        )
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
        
        self.fc1 = nn.Linear(in_features=CN_params['l_fc1']['in'], out_features=CN_params['l_fc1']['out'])
        self.fc2 = nn.Linear(CN_params['l_fc2']['in'], CN_params['l_fc2']['out'])

        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.AdamW(
        #     self.parameters(), 
        #     lr=LR_params['etas'][0],
        #     weight_decay=GD_params['lam']
        # )
        self.optimizer = optim.SGD(
            self.parameters(), 
            lr=LR_params['etas'][0],
            momentum=0.9,
            weight_decay=GD_params['lam']
        )
        
        self.trainloader, self.testloader, self.classes = loadData(batch_size=GD_params['n_batch'])
        train_dataset, val_dataset = random_split(
            self.trainloader.dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        self.trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.valloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    def forward(self, x):
        x = self.patchify(x)
        
        # ========================
        # VGG Block-1
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        # Apply maxpooling layer
        x = self.pool1(x)
        # ========================

        # Flattening (to connect to fc1 layer)
        x = x.view(x.size(0), -1)

        # Run through fc1
        x = self.fc1(x)
        x = F.relu(x)

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
        return accuracy, loss


    def trainModel(self, debug=True, plot=False):
        """
        Train model using CLR with increasing cycle lengths
        """
        n_cycles = self.GD_params['n_cycles']
        current_ns = (self.GD_params['n_epochs'] * len(self.trainloader)) // 2
        
        if plot:
            steps = 0
            results = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [], 'steps': []}
        
        for cycle in range(n_cycles):
            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer, 
                base_lr=self.LR_params['etas'][0],
                max_lr=self.LR_params['etas'][1],
                step_size_up=current_ns,
                mode='triangular'
            )
            
            steps_in_current_cycle = 2*current_ns
            steps_made_in_current_cycle = 0
            epochs_in_cycle = (2 ** cycle) * self.GD_params['n_epochs']
            epochs_made_in_cycle = 0
            
            while steps_made_in_current_cycle < steps_in_current_cycle:
                epochs_made_in_cycle += 1
                running_loss = 0.0
                batch_count = 0
                
                for i, (inputs, labels) in enumerate(self.trainloader, 0):
                    self.optimizer.zero_grad()

                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    steps_made_in_current_cycle += 1
                    running_loss += loss.item()
                    batch_count += 1
                    
                    if steps_made_in_current_cycle >= steps_in_current_cycle:
                        break
                    

                if debug:
                    avg_loss = running_loss / batch_count
                    val_accuracy, val_loss = self.evaluate(self.valloader)
                    print(f'Cycle {cycle+1}/{self.GD_params['n_cycles']}, Epoch {epochs_made_in_cycle}/{epochs_in_cycle} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}')
                if plot:
                    results['loss'].append(avg_loss)
                    results['val_loss'].append(val_loss)
                    results['acc'].append(self.evaluate(self.trainloader)[0])
                    results['val_acc'].append(val_accuracy)
                    results['steps'].append(steps)
                    steps += 1
                
            current_ns *= 2      
            
        if plot:
            plotPerformance(results) 
