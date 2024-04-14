import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.Vector2 import Vector2

# Constants
L = 10.0  # Size of the box
N = 100   # Number of water molecules
dt = 0.01 # Time step
steps = 1000 # Number of simulation steps
decay = 0.8
epsilon = 1.0  # Lennard-Jones potential depth
sigma = 1.0    # Lennard-Jones potential length scale

smoothRadius = 2

# Initialize positions and velocities randomly
positions = []
velocities = []

for i in range(N):
    positions.append(Vector2(2 * np.random.rand() - 1, 2 * np.random.rand() - 1))
    velocities.append(Vector2(0,0))

x_values = [vector.x for vector in positions]
y_values = [vector.y for vector in positions]
# Create a figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlim(-L - 2, L + 2)
ax.set_ylim(-L - 2, L + 2)
scat = ax.scatter(x_values, y_values)




def applyGravity():
    global positions, velocities
    
    # Add Gravity
    for i in range(N):
        velocities[i].y -= 9.8

def resolveCollisions():
    global positions, velocities

    for i in range(N):
        if positions[i].x > L:
            positions[i].x = L - 0.1
            velocities[i].x *= -decay
        elif positions[i].x < -L:
            positions[i].x = -L + 0.1
            velocities[i].x *= -decay

        if positions[i].y > L:
            positions[i].y = L - 0.1
            velocities[i].y *= -decay
        elif positions[i].y < -L:
            positions[i].y = -L + 0.1
            velocities[i].y *= -decay

# Update function for the animation
def update(frame):
    global positions, velocities
    
    # Update positions
    for i in range(N):
        positions[i] += velocities[i] * dt
            
    # Apply boundary conditions
    resolveCollisions()

    applyGravity()

    x_values = np.array([vector.x for vector in positions])
    y_values = np.array([vector.y for vector in positions])

    # Create a 2D array-like object for the offsets
    offsets = np.column_stack((x_values, y_values))

    # Set the offsets
    scat.set_offsets(offsets)

# Create animation
ani = FuncAnimation(fig, update, frames=steps, interval=50)

plt.show()
