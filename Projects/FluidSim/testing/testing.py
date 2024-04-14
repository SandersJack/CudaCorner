import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
L = 10.0  # Size of the box
N = 100   # Number of water molecules
dt = 0.01 # Time step
steps = 1000 # Number of simulation steps
decay = 0.8
epsilon = 1.0  # Lennard-Jones potential depth
sigma = 1.0    # Lennard-Jones potential length scale

# Initialize positions and velocities randomly
positions = 2 * np.random.rand(N, 2) - 1
velocities = np.zeros([N, 2])

# Create a figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlim(-L - 2, L + 2)
ax.set_ylim(-L - 2, L + 2)
scat = ax.scatter(positions[:,0], positions[:,1])


def applyGravity():
    global positions, velocities
    
    # Add Gravity
    for i in range(N):
        velocities[i, 1] -= 9.8

def resolveCollisions():
    global positions, velocities

    for i in range(N):
        for j in range(2):
            if positions[i, j] > L:
                positions[i, j] = L - 0.1
                velocities[i, j] *= -decay
            elif positions[i, j] < -L:
                positions[i, j] = - L + 0.1
                velocities[i, j] *= -decay

# Update function for the animation
def update(frame):
    global positions, velocities
    
    # Update positions
    positions += velocities * dt
    
    # Apply boundary conditions
    resolveCollisions()

    applyGravity()

    # Update scatter plot
    scat.set_offsets(positions)

# Create animation
ani = FuncAnimation(fig, update, frames=steps, interval=50)

plt.show()
