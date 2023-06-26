import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Define the custom dataset
class SpamDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.Tensor(self.data[idx])
        target = torch.Tensor([self.targets[idx]])
        return data, target

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the dataset and preprocess it
data = pd.read_csv('spam_or_not_spam.csv')

# Drop rows with missing values
data = data.dropna()

X = data['email'].values
y = data['label'].values

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create custom dataset and data loaders
train_dataset = SpamDataset(X_train, y_train)
test_dataset = SpamDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model
input_dim = X_train.shape[1]
model = Net(input_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * inputs.size(0)
    
    # Calculate average epoch loss
    epoch_loss /= len(train_dataset)
    train_losses.append(epoch_loss)
    
    # Print training progress
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')

# Save the trained model
torch.save(model.state_dict(), 'spam_detection_model.pt')
print('Model saved successfully.')

# Evaluate the model
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted_labels = (outputs > 0.5).squeeze().long()
        total_correct += (predicted_labels == targets.squeeze().long()).sum().item()
        total_samples += targets.size(0)

accuracy = total_correct / total_samples
print(f'Test Accuracy: {accuracy * 100}%')

# Plot and save the loss over time
plt.plot(range(1, num_epochs+1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.savefig('loss_plot.png')
plt.show()
