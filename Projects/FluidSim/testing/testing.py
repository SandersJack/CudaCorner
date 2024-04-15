import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import multiprocessing
from src.Vector2 import Vector2

# Constants
L = 10.0  # Size of the box
N = 500   # Number of water molecules
dt = 0.01 # Time step
steps = 1000 # Number of simulation steps
decay = 0.8
epsilon = 1.0  # Lennard-Jones potential depth
sigma = 1.0    # Lennard-Jones potential length scale
mass = 1
smoothRadius = 1.2

targetDensity = 2.75
pressureMultiplier = 100

# Initialize positions and velocities randomly
positions = []
velocities = []
densities = N * [1]

for i in range(N):
    positions.append(Vector2(L * np.random.rand() - L/2, L * np.random.rand() - L/2))
    velocities.append(Vector2(0,0))

x_values = [vector.x for vector in positions]
y_values = [vector.y for vector in positions]
# Create a figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlim(-L - 2, L + 2)
ax.set_ylim(-L - 2, L + 2)
scat = ax.scatter(x_values, y_values)


def smoothingKernal(radius, dis):
    volume = np.pi * radius**8 / 4
    val = max(0, radius * radius - dis * dis)
    return val * val * val / volume

def smoothingKernalDerivative(radius, dis):
    if( dis >= radius):
        return 0
    f = radius * radius - dis * dis
    scale = -24 / (np.pi * radius**8)
    return scale * dis * f * f

def smoothingKernal_new(radius, dis):
    if( dis >= radius): return 0
    
    volume = (np.pi * radius**4) / 6
    return (radius - dis) * (radius - dis) / volume 

def smoothingKernalDerivative_new(radius, dis):
    if(dis >= radius): return 0

    scale = 12 / (radius**4 * np.pi)
    return (dis - radius) * radius

def calculateDensity(samplePoint):
    density = 0
    mass  = 1

    for point in positions:
        dist = (point - samplePoint).magnitude()
        influence = smoothingKernal_new(smoothRadius, dist)
        density += mass * influence
    
    return density

def calculateDensities():
    global densities
    densities = [calculateDensity(samplePoint) for samplePoint in positions]

def convertDensityToPressure(density):
    densityError = density - targetDensity
    pressure = densityError * pressureMultiplier
    return pressure

def GetRandomDir():
    # Generate random x and y components
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    
    # Normalize the vector to get a unit direction vector
    magnitude = (x ** 2 + y ** 2) ** 0.5
    if magnitude != 0:
        x /= magnitude
        y /= magnitude
    
    return Vector2(x, y)

def calculateSharedPressure(densityA, densityB):
    pressureA = convertDensityToPressure(densityA)
    pressureB = convertDensityToPressure(densityB)
    return (pressureA + pressureB) / 2


def calculatePressureForce(args):
    particleIndex, positions, densities = args
    densityGradient = Vector2()

    for i in range(N):
        if (particleIndex == i): continue

        offset = (positions[i] - positions[particleIndex])
        dist = offset.magnitude()
        dir = GetRandomDir() if dist == 0 else offset / dist
        slope = smoothingKernalDerivative_new(smoothRadius, dist)
        density = densities[i]
        sharedPressure = calculateSharedPressure(density, densities[particleIndex])
        densityGradient += sharedPressure * dir * slope * mass / density

    return densityGradient

def calculateDensityForces():
    global velocities

    for i in range(N):
        pressureForce = calculatePressureForce((i, positions, densities))
        pressureAccel = pressureForce / densities[i]
        velocities[i] += pressureAccel * dt



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

    calculateDensities()
    calculateDensityForces()


    #applyGravity()

    resolveCollisions()

    x_values = np.array([vector.x for vector in positions])
    y_values = np.array([vector.y for vector in positions])

    # Create a 2D array-like object for the offsets
    offsets = np.column_stack((x_values, y_values))

    # Set the offsets
    scat.set_offsets(offsets)

    print("step")

# Create animation
ani = FuncAnimation(fig, update, frames=steps)

plt.show()
