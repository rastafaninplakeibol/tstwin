import matplotlib.pyplot as plt

# Read data from files
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data

# File paths
training_loss_file = 'lstm_training_loss.log'
error_file = 'lstm_error.log'

# Read data
training_loss = read_data(training_loss_file)
error = read_data(error_file)

# Plot data
plt.figure(figsize=(10, 5))

plt.plot(training_loss, label='Training Loss')
plt.plot(error, label='Error')

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Error over Epochs')
plt.legend()

plt.savefig('training_loss_error.png')
#plt.show()