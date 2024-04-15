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
smoothRadius = 1

targetDensity = 2.75
pressureMultiplier = 1000

# Initialize positions and velocities randomly
positions = []
velocities = []
densities = N * [1]

spatialLookup = np.ones(N)
startIndex = np.ones(N)

cellOffsets = [[-1, 1],
	[0, 1],
	[1, 1],
	[-1, 0],
	[0, 0],
	[1, 0],
	[-1, -1],
	[0, -1],
	[1, -1]]

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


def calculatePressureForce(particleIndex):
    global densities, positions

    return EachPointWithinRadius(particleIndex)

def calculateDensityForces():
    global velocities

    for i in range(N):
        pressureForce = calculatePressureForce(i)
        pressureAccel = pressureForce / densities[i]
        velocities[i] += pressureAccel * dt



def applyGravity():
    global positions, velocities
    
    # Add Gravity
    for i in range(N):
        velocities[i].y -= 1

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

def PosToCellCoord(point, radius):
    cellX = int(point.x / radius)
    cellY = int(point.y / radius)
    return (cellX, cellY)

def HashCell(cellX, cellY):
    a = cellX * 15823
    b = cellY * 9737333
    return a + b

def GetKeyFromHash(hash):
    global spatialLookup
    return int(hash) % int(np.size(spatialLookup))


def UpdateSpatialLookup():
    global spatialLookup, positions
    for i in range(len(positions)):
        cellX , cellY = PosToCellCoord(positions[i], smoothRadius)
        cellKey = GetKeyFromHash(HashCell(cellX, cellY))
        spatialLookup[i] = cellKey
        startIndex[i] = -99

    spatialLookup = np.sort(spatialLookup)

    for t in range(len(positions)):
        key = int(spatialLookup[t])
        keyPrev = -99 if t == 0 else spatialLookup[t - 1]
        if(key != keyPrev):
            startIndex[key] = t


def EachPointWithinRadius(sampleIndex):
    global positions, densities
    samplePoint = positions[sampleIndex]
    centerX, centerY = PosToCellCoord(samplePoint, smoothRadius)
    sqrtRadius = smoothRadius * smoothRadius

    pressureForce = Vector2()

    for i in range(len(cellOffsets)):

        key = GetKeyFromHash(HashCell(centerX + cellOffsets[i][0], centerY + cellOffsets[i][1]))
        cellStartIndex = startIndex[key]

        for t in range(int(cellStartIndex), np.size(spatialLookup)):
            if spatialLookup[t] != key: break

            particleIndex = t
            sqrDist = ((positions[particleIndex] - samplePoint).magnitude())**2

            if (sqrDist <= sqrtRadius):
                #print("Boom")
                # Do somthing
                if (particleIndex == sampleIndex): continue

                offset = (positions[particleIndex] - samplePoint)
                dist = offset.magnitude()
                dir = GetRandomDir() if dist == 0 else offset / dist
                slope = smoothingKernalDerivative_new(smoothRadius, dist)
                density = densities[particleIndex]
                
                sharedPressure = calculateSharedPressure(density, densities[particleIndex])
                pressureForce += sharedPressure * dir * slope * mass / density
    #print(pressureForce)
    return pressureForce


# Update function for the animation
def update(frame):
    global positions, velocities
    
    # Update positions
    for i in range(N):
        positions[i] += velocities[i] * dt


    UpdateSpatialLookup()

    calculateDensities()
    calculateDensityForces()


    applyGravity()

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
