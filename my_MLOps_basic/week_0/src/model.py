import torch
from torch.utils.data import DataLoader, TensorDataset
from data_ import create_dataset  # Import the dataset file

import torch.nn as nn
import torch.optim as optim


X, y = create_dataset()
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),  # Input features: 2, Hidden neurons: 16
            nn.ReLU(),
            nn.Linear(16, 1),  # Output neurons: 1
            nn.Sigmoid()  # Sigmoid for binary classification
        )
    
    def forward(self, x):
        return self.fc(x)

class ModelHandler:
    def __init__(self):
        # Initialize model, criterion, and optimizer
        self.model = SimpleClassifier()
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    
    def train(self, dataloader, epochs=20):
        for epoch in range(epochs):  # Train for a number of epochs
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X).squeeze()  # Forward pass
                loss = self.criterion(outputs, batch_y.float())  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
            
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    def test(self, input_data):
        with torch.no_grad():
            prediction = self.model(input_data)
            return prediction.item()

def main():

    X, y = create_dataset()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # Binary labels (0 or 1)

    # Initialize ModelHandler and train the model
    handler = ModelHandler()
    handler.train(dataloader)
    
    # Test the model
    test_input = torch.tensor([[0.5, 0.5]])
    prediction = handler.test(test_input)
    print(f"Prediction for input {test_input}: {prediction:.4f}")

if __name__ == "__main__":
    print("model_training_started")
    main()
