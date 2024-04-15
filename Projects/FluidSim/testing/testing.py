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
    ### Loop over all points and calculate the density
    for i in range(N):
        pressureForce = calculatePressureForce(i)
        pressureAccel = pressureForce / densities[i]
        velocities[i] += pressureAccel * dt



def applyGravity():
    global positions, velocities
    ### Loop over all particles and add Gravity
    for i in range(N):
        velocities[i].y -= 1

def resolveCollisions():
    global positions, velocities
    ### Loop over all positions
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
    ### Each cell is smoothing radius size
    cellX = int(point.x / radius)
    cellY = int(point.y / radius)
    return (cellX, cellY)

def HashCell(cellX, cellY):
    ### Encode the position within a hash
    a = cellX * 15823
    b = cellY * 9737333
    return a + b

def GetKeyFromHash(hash):
    global spatialLookup
    ### Divide the hash by the size of the lookuptable to get a key
    return int(hash) % int(np.size(spatialLookup))


def UpdateSpatialLookup():
    global spatialLookup, positions
    ### Loop over all points
    for i in range(len(positions)):
        ### Calculate the Position to Cell Cord of ponit
        cellX , cellY = PosToCellCoord(positions[i], smoothRadius)
        ### Hash the position then get a key from the hash
        cellKey = GetKeyFromHash(HashCell(cellX, cellY))
        ### Save this key in the point position
        spatialLookup[i] = cellKey
        ### Set a non value for the index
        startIndex[i] = -99

    ### Sort the array
    spatialLookup = np.sort(spatialLookup)

    ### Loop again over all positions
    for t in range(len(positions)):
        ### Find the cellKey at this position
        key = int(spatialLookup[t])
        keyPrev = -99 if t == 0 else spatialLookup[t - 1]
        ## If the key is not the same as the previous, set the start Index as this denotes the first point in this cell
        if(key != keyPrev):
            startIndex[key] = t


def EachPointWithinRadius(sampleIndex):
    global positions, densities
    ### Get the sample point
    samplePoint = positions[sampleIndex]
    ### Get the Cell value of the sample pont
    centerX, centerY = PosToCellCoord(samplePoint, smoothRadius)
    ### Get the square of the smoothing radius
    sqrtRadius = smoothRadius * smoothRadius

    ### Set emtpy pressure Force
    pressureForce = Vector2()

    ### Loop over the cell offsets
    for i in range(len(cellOffsets)):
        ### Get the key of the adjacent/same search cell
        key = GetKeyFromHash(HashCell(centerX + cellOffsets[i][0], centerY + cellOffsets[i][1]))
        ### Get the start index for the first point in that cell
        cellStartIndex = startIndex[key]

        ### Loop from that start index to the end of the lookup
        for t in range(int(cellStartIndex), np.size(spatialLookup)):
            ### If te lookup value is the same as the key break
            if spatialLookup[t] != key: break

            ### Set the particle index
            particleIndex = t
            ### Calculate the square distance between the sample point and this particle in the test cell
            sqrDist = ((positions[particleIndex] - samplePoint).magnitude())**2

            ### If it is smaller that smoothing radius then continue on
            if (sqrDist <= sqrtRadius):
                # Do somthing
                ### If the index is the same as the sample, skip as it is the same point
                if (particleIndex == sampleIndex): continue

                ### Calculate the offset between the two points
                offset = (positions[particleIndex] - samplePoint)
                ### Get the magnitude of this offset
                dist = offset.magnitude()
                ### Get the direction vector of between the two points, if there the same dist == 0 just choose a random dir
                dir = GetRandomDir() if dist == 0 else offset / dist
                ### Calculate the slope of the smoothing function
                slope = smoothingKernalDerivative_new(smoothRadius, dist)
                ### Get the point density
                density = densities[particleIndex]
                ### Calculate the two points shared pressure
                sharedPressure = calculateSharedPressure(density, densities[particleIndex])
                ### Calculate the pressure force
                pressureForce += sharedPressure * dir * slope * mass / density
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
