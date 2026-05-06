import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataHelper import loadData

class Network(nn.Module):
    def __init__(self, LR_params, GD_params, CN_params):
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
        
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=CN_params['n_f'], 
            kernel_size=CN_params['f'], 
            stride=CN_params['f']
        )
        
        self.fc1 = nn.Linear(CN_params['n_f'] * (32 // CN_params['f'])**2, GD_params['n_hidden'])
        self.fc2 = nn.Linear(GD_params['n_hidden'], GD_params['k'])  

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.parameters(), 
            lr=LR_params['etas'][0],
            momentum=0.9,
            weight_decay=GD_params['lam']
        )
        # TODO: Test adam later
        # self.optimizer = optim.Adam(self.parameters(), lr=LR_params['etas'][0])
        self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer, 
                base_lr=self.LR_params['etas'][0],
                max_lr=self.LR_params['etas'][1],
                step_size_up=CN_params['n_s'],
                mode='triangular'
            )
        
        self.trainloader, self.testloader, self.classes = loadData(batch_size=GD_params['n_batch'])


    def forward(self, x):
        """
        self: Network instance
        x: Input data to be passed through the network
        """
        # Pass through conv layer
        x = self.conv1(x)
        
        # Apply relu
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        # Run through fc1
        x = self.fc1(x)

        # Second relu
        x = F.relu(x)

        # fc2 layer
        x = self.fc2(x)

        # Softmax, convert to probabilities
        # x = F.softmax(x, dim=1)
        return x
    
    def loss(self):
        pass

    def evaluate(self):
        """
        Evaluate the trained network on the trained parameters

        dataloader: The specified dataloader, either training, 
                    valuation or test set.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy


    def trainModel(self):
        n_cycles = self.GD_params['n_cycles']
        current_ns = (self.GD_params['n_epochs'] * len(self.trainloader)) // 2
        
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
                
                avg_loss = running_loss / batch_count      
                print(f'Cycle {cycle+1}/{self.GD_params['n_cycles']}, Epoch {epochs_made_in_cycle}/{epochs_in_cycle} | Loss: {avg_loss:.4f}') 
                
            current_ns *= 2            
