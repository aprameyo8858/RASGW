import sys
import os
import numpy as np
import torch
sys.path.append('./lib')
from risgw_original import risgw_gpu_original
import matplotlib.pyplot as pl
from risgw import risgw_gpu
from sgw_pytorch import sgw_gpu
from sgw_pytorch_original import sgw_gpu_original
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data as data_t
import torch.optim as optim
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import torch.optim as optim

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cuda'

class StopError(Exception):
    pass

def create2D_ds(n_samples):
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    X, y = noisy_circles
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X, y

def create3D_ds(n_samples):
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    X, y = noisy_circles
    X = np.column_stack((X, np.repeat(1, X.shape[0])))
    
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X[y == 1, 2] = X[y == 1, 2] + 2
    return X, y

def ds_to_torch(X, y,device):
    return torch.FloatTensor(X).to('cuda'), torch.LongTensor(y).to('cuda')

#N = 150
#N_sup = 50  # number of supervised points
#K = 2  # number of classes
#X2D, y2D = create2D_ds(n_samples=N)
#X3D, y3D = create3D_ds(N)

#X2D_torch, y2D_torch = ds_to_torch(X2D, y2D,device)
#X3D_torch, y3D_torch = ds_to_torch(X3D, y3D,device)

#X2D_torch=X2D_torch.to('cuda')
#X3D_torch=X3D_torch.to('cuda')

# 3D plotting
#fig = pl.figure()
#ax = fig.add_subplot(111, projection='3d')
#colors = {0: 'r', 1: 'b', 2: 'k', 3: 'g'}
#ax.scatter(np.array(X3D)[:, 0], np.array(X3D)[:, 1], np.array(X3D)[:, 2], c=[colors[y3D[i]] for i in range(len(y3D))])

# 2D plotting (use a separate z-value for visibility in 3D)
#ax.scatter(np.array(X2D)[:, 0], np.array(X2D)[:, 1], -10, c=[colors[y2D[i]] for i in range(len(y2D))])

# Save the plot to a file (for example, "3d_plot.png")
#plot_filename = "plots/3d_plot.png"  # Specify the file path
#pl.savefig(plot_filename)

# Close the figure to free up memory (important if generating many plots)
#pl.close()

#print(f"Plot saved as {plot_filename}")

# Generate a 2D spiral dataset
def create2D_ds_spiral(n_samples):
    """Create a 2D spiral dataset."""
    theta = np.linspace(0, 4 * np.pi, n_samples)  # Spiral angle
    r = np.linspace(0, 1, n_samples)  # Radius
    x = r * np.sin(theta)  # x-coordinate
    y = r * np.cos(theta)  # y-coordinate
    
    # Labels: Use theta to split into two classes
    y_labels = (theta > 2 * np.pi).astype(int)  # Class 1 for θ > π and Class 0 for θ < π
    
    X = np.column_stack((x, y))
    X = (X - np.min(X)) / (np.max(X) - np.min(X))  # Normalize the dataset
    
    return X, y_labels

# Generate a 3D spiral dataset
def create3D_ds_spiral(n_samples):
    """Create a 3D spiral dataset."""
    theta = np.linspace(0, 4 * np.pi, n_samples)  # Spiral angle
    r = np.linspace(0, 1, n_samples)  # Radius
    z = np.linspace(0, 1, n_samples)  # Z coordinate
    
    x = r * np.sin(theta)  # x-coordinate
    y = r * np.cos(theta)  # y-coordinate
    
    # Adding noise to simulate real-world data
    noise = np.random.normal(scale=0.1, size=(n_samples, 3))  # Noise for 3D
    
    X = np.column_stack((x, y, z)) + noise  # 3D spiral with noise
    
    # Labels: Use theta to split into two classes
    y_labels = (theta > 2 * np.pi).astype(int)  # Class 1 for θ > π and Class 0 for θ < π
    
    # Normalize the dataset
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    
    return X, y_labels

# Plotting the 2D spiral dataset
def plot_2d_spiral(X, y):
    """Visualize the 2D spiral dataset."""
    pl.figure(figsize=(6, 6))
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.RdYlBu, s=20)
    pl.title("2D Spiral Dataset")
    pl.xlabel("X")
    pl.ylabel("Y")
    pl.colorbar(label='Class')

    # Save the plot to a file
    plot_filename = "plots/2d_spiral.png"
    pl.savefig(plot_filename)
    
    # Close the figure to free up memory
    pl.close()

    print(f"Plot saved as {plot_filename}")

# Plotting the 3D spiral dataset
def plot_3d_spiral(X, y):
    """Visualize the 3D spiral dataset."""
    fig = pl.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=pl.cm.RdYlBu, s=20)
    ax.set_title("3D Spiral Dataset")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(scatter, label='Class')

    # Save the plot to a file
    plot_filename = "plots/3d_spiral.png"
    pl.savefig(plot_filename)
    
    # Close the figure to free up memory
    pl.close()

    print(f"Plot saved as {plot_filename}")

# Example usage
N = 1500
N_sup = 500  # number of supervised points
K = 1  # number of classes it was 2 initially
#N = 150  # Number of samples
X2D, y2D = create2D_ds_spiral(n_samples=N)
X3D, y3D = create3D_ds_spiral(N)

X2D_torch, y2D_torch = ds_to_torch(X2D, y2D,device)
X3D_torch, y3D_torch = ds_to_torch(X3D, y3D,device)

X2D_torch=X2D_torch.to('cuda')
X3D_torch=X3D_torch.to('cuda')
# Plot and save the 2D Spiral Dataset
colors = {0: 'r', 1: 'b', 2: 'k', 3: 'g'}
plot_2d_spiral(X2D, y2D)

# Plot and save the 3D Spiral Dataset
plot_3d_spiral(X3D, y3D)


class Target_model(nn.Module):
    def __init__(self):
        super(Target_model, self).__init__()
        self.fc1 = nn.Linear(2, 256)   #originally it was this
        #self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.fc4 = nn.Linear(3, K)
    def forward(self, x):
        return self.forward_remaining(self.forward_partial(x))
    def forward_partial(self, x):
        x_init = x.clone()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.cat((x_init,x),dim=1)
        #x = F.relu(x)        # was commented out originally
        return x.to('cuda')
    def forward_remaining(self,x):
        #x = F.relu(self.fc1(x))            #was commented out originally
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
      
# initialization function, first checks the module type,
# then applies the desired changes to the weights
def init_normal(m):
  if type(m) == nn.Linear:
    nn.init.normal_(m.weight, mean=0, std=0.2)


target_model=Target_model()
target_model.apply(init_normal)

# Ensure directories exist for saving plots
if not os.path.exists("plots/rasgw"):
    os.makedirs("plots/rasgw")

if not os.path.exists("plots/sgw"):
    os.makedirs("plots/sgw")

# Define the optimizer and model
#optimizer = optim.SGD(target_model.parameters(), lr=5e-3, momentum=0.9)    #
optimizer = optim.Adam(target_model.parameters(), lr=2e-3, betas=(0.5, 0.99))

# Loss lists for both methods
losses_rasgw = []  # Losses for RASGW
losses_sgw = []    # Losses for SGW

# Ensure the model is on the correct device (GPU)
target_model = Target_model().to('cuda')  

def generate_random_noise(batch_size, noise_dim=2):
    """ Generate random noise Z of shape (batch_size, noise_dim) for the source distribution """
    Z = torch.randn(batch_size, noise_dim).to(device)  # Gaussian noise (source)
    return Z

# Training for SGW
for epoch in range(3000):        #it was 3000
    # Generate random noise Z (sampled from a Gaussian distribution)
    Z = generate_random_noise(X2D_torch.size(0), noise_dim=2)  # Adjust noise_dim if needed
    
    # Pass the random noise Z through the target model (generator)
    Xt = target_model.forward_partial(Z)  # Xt is the generated data from the random noise
    
    #Xt = target_model.forward_partial(X2D_torch)
    Xs = X3D_torch
    loss_, log = risgw_gpu_original(Xs.to(device), Xt.to(device), device, nproj=500, max_iter=100, tolog=True, retain_graph=True)      #nproj=50 originally      
    Delta = log['Delta']
    loss = sgw_gpu_original(Xs.matmul(Delta.detach()).to(device), Xt.to(device), device, nproj=500)                #nproj=50 originally
    #loss = sgw_gpu_original(Xs.to(device), Xt.to(device), device, nproj=50)
    #print("SGW, Epoch,loss:",epoch,loss_.item())
    print("SGW, Epoch,loss:",epoch,loss)
    #loss=loss_
    #losses_sgw.append(loss_)
    #loss_=torch.tensor(loss_)
    losses_sgw.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        with torch.no_grad():
            # Generate new data using random noise Z
            Z = generate_random_noise(X2D_torch.size(0), noise_dim=2)  # Generating noise for the batch
            Xs_new = target_model.forward_partial(Z).clone().detach().cpu().numpy()  # Generate data from noise
            #Xs_new = target_model.forward_partial(X2D_torch).clone().detach().cpu().numpy()  # Generate data from noise
        
            # Visualize generated data vs actual target data
            fig = pl.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            # 1. Plot the target data in 2D (X2D) as a scatter plot in 3D space with a fixed Z-axis for visibility
            #ax.scatter(np.array(X2D)[:, 0], np.array(X2D)[:, 1], np.zeros(len(X2D)), c=[colors[y2D[i]] for i in range(len(y2D))], label="Target Data (2D)")

            # 2. Plot the generated data (Xs_new) in 3D
            #ax.scatter(Xs_new[:, 0], Xs_new[:, 1], Xs_new[:, 2], c='k', label="Generated Data (3D)")
            #ax.scatter(np.array(X3D)[:, 0], np.array(X3D)[:, 1], np.array(X3D)[:, 2], c=[colors[y3D[i]] for i in range(len(y3D))])  # Actual target data
            #ax.scatter(Xs_new[:, 0], Xs_new[:, 1], Xs_new[:, 2], c='k')  # Generated data
            ax.scatter(Xs_new[:, 0], Xs_new[:, 1], Xs_new[:, 2], c=[colors[y3D[i]] for i in range(len(y3D))])  # Generated data
        
            plot_filename = f"plots/sgw/epoch_{epoch}_3dscatter_spiral.png"
            pl.savefig(plot_filename)
            pl.close()
            
# Training for RASGW
for epoch in range(-1):
    Xt = target_model.forward_partial(X2D_torch.to('cuda'))
    Xs = X3D_torch
    loss_, log = risgw_gpu(Xs.to(device), Xt.to(device), device, nproj=50, max_iter=100, tolog=True, retain_graph=True)
    Delta = log['Delta']
    loss = sgw_gpu(Xs.matmul(Delta.detach()).to(device), Xt.to(device), device, nproj=50)
    print("RASGW, Epoch,loss:",epoch,loss.item())
    
    losses_rasgw.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        with torch.no_grad():
            Xs_new = target_model.forward_partial(X2D_torch).clone().detach().cpu().numpy()
            fig = pl.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            colors = {0: 'r', 1: 'b', 2: 'k', 3: 'g'}
            ax.scatter(np.array(X3D)[:, 0], np.array(X3D)[:, 1], np.array(X3D)[:, 2],
                       c=[colors[y3D[i]] for i in range(len(y3D))])
            ax.scatter(Xs_new[:, 0], Xs_new[:, 1], Xs_new[:, 2], c='k')
            plot_filename = f"plots/rasgw/epoch_{epoch}_3dscatter.png"
            pl.savefig(plot_filename)
            pl.close()  # Close to avoid memory issues


# Now, after both trainings, plot the individual and combined loss curves

# Create a new figure for the RASGW loss plot
pl.figure(figsize=(10, 6))
pl.plot(losses_rasgw, label='RASGW Loss', color='r', lw=2)
pl.title('Loss along iterations for RASGW')
pl.xlabel('Epochs')
pl.ylabel('Loss')
pl.legend()
# Save the individual RASGW loss plot
pl.savefig("plots/rasgw/rasgw_loss_plot.png")
pl.close()

# Create a new figure for the SGW loss plot
pl.figure(figsize=(10, 6))
pl.plot(losses_sgw, label='SGW Loss', color='b', lw=2)
pl.title('Loss along iterations for SGW')
pl.xlabel('Epochs')
pl.ylabel('Loss')
pl.legend()
# Save the individual SGW loss plot
pl.savefig("plots/sgw/sgw_loss_plot.png")
pl.close()

# Create a combined loss plot for both RASGW and SGW
pl.figure(figsize=(10, 6))
pl.plot(losses_rasgw, label='RASGW Loss', color='r', lw=2)
pl.plot(losses_sgw, label='SGW Loss', color='b', lw=2)
pl.title('Loss along iterations for RASGW and SGW')
pl.xlabel('Epochs')
pl.ylabel('Loss')
pl.legend()
# Save the combined loss plot
pl.savefig("plots/combined_loss_plot.png")
pl.close()

print("All plots have been saved.")
