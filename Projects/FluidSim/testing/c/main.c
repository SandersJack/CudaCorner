
#include <SDL2/SDL.h>  
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h> 
#include <time.h> 
#include <math.h>

#define M_PI 3.14159265358979323846

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define NUM_PARTICLES 1000


#define DECAY 0.5
#define MASS 1
#define SMOOTHRADIUS 100
#define TARGET_DENSITY 0.002 //0.021
#define PRESSURE_MULT 50
#define VISCOSITY 10

#define DT 0.01

int frameCount = 0;
float fps = 0;
Uint32 lastFrameTime = 0;

void calculateFPS(Uint32 currentTime) {
    // Calculate time elapsed since the last frame update
    float elapsedTime = (currentTime - lastFrameTime) / 1000.0f; // Convert to seconds

    // Update frame count
    frameCount++;

    // If one second has passed, calculate FPS
    if (elapsedTime >= 1.0f) {
        fps = frameCount / elapsedTime;
        frameCount = 0;
        lastFrameTime = currentTime;
    }
}

int L = WINDOW_WIDTH - 50;

typedef struct {
    float x;
    float y; 
    float pred_x;
    float pred_y;
    float dx;
    float dy;
} Particle;

typedef struct {
    float first;
    float second;
} FloatPair;

Particle particles[NUM_PARTICLES];

float densities[NUM_PARTICLES];

int spatialLookup[NUM_PARTICLES];
int startIndex[NUM_PARTICLES];

int cellOffsets[9][2] = {{-1, 1},
	{0, 1},
	{1, 1},
	{-1, 0},
	{0, 0},
	{1, 0},
	{-1, -1},
	{0, -1},
	{1, -1}};

int compare(const void *a, const void *b) {
    const float *x = (const float *)a;
    const float *y = (const float *)b;
    if (*x < *y) return -1;
    else if (*x > *y) return 1;
    else return 0;
}

float GetRandomDir() {
    // Generate a random float between -1 and 1
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

int getKeyFromHash(int hash){
    int sizeOfSL = sizeof(spatialLookup) / sizeof(spatialLookup[0]);
    return hash % sizeOfSL;
}

int hashCell(int cellX, int cellY){
    int a = cellX * 15823;
    int b = cellY * 9737333;
    return a + b;
}

float smoothingKernal_new(float radius, float dis){
    if( dis >= radius) return 0;
    
    float volume = (M_PI * pow(radius, 4)) / 6;
    return (radius - dis) * (radius - dis) / volume;
}

float smoothingKernalDerivative_new(float radius, float dis){
    if(dis >= radius) return 0;

    float scale = 12 / (pow(radius,4) * M_PI);
    return (dis - radius) * scale;
}


FloatPair posToCellCoord(Particle particle, float radius){
    FloatPair pair;
    pair.first = (int)(particle.pred_x / radius);
    pair.second = (int)(particle.pred_y / radius);
    return pair;
}

float convertDensityToPressure(float density){
    float densityError = density / TARGET_DENSITY;
    float diff = density - TARGET_DENSITY;
    if (diff < 0) densityError *= -1;

    float pressure =  densityError * PRESSURE_MULT;
    //printf("CAl Pressure: %f %f %f \n", density, densityError, pressure);
    return pressure;
}

float calculateSharedPressure(float densityA, float densityB){
    float pressureA = convertDensityToPressure(densityA);
    float pressureB = convertDensityToPressure(densityB);
    return (pressureA + pressureB) / 2;
}

FloatPair calculateViscosityForce(int sampleIndex){
    Particle sampleParticle = particles[sampleIndex];

    FloatPair cell = posToCellCoord(sampleParticle, SMOOTHRADIUS);
    float centerX = cell.first;
    float centerY = cell.second;

    float sqRadius = SMOOTHRADIUS * SMOOTHRADIUS;

    FloatPair viscosityForce;
    viscosityForce.first = 0;
    viscosityForce.second = 0;

    for(int i=0;i<9; i++){
        int key = getKeyFromHash(hashCell(centerX + cellOffsets[i][0], centerY + cellOffsets[i][1]));
        
        int cellStartIndex = startIndex[key];
        
        int sizeOfSL = sizeof(spatialLookup) / sizeof(spatialLookup[0]);

        for(int t=cellStartIndex; t<sizeOfSL; t++){

            if(spatialLookup[t] != key) break;

            int particleIndex = t;
                float dist_X = particles[particleIndex].pred_x - sampleParticle.pred_x;
                float dist_Y = particles[particleIndex].pred_y - sampleParticle.pred_y;
                float dist = sqrt(dist_X*dist_X + dist_Y*dist_Y); 

                float infuence = smoothingKernal_new(SMOOTHRADIUS, dist);

                viscosityForce.first += (particles[t].dx - particles[sampleIndex].dx) * infuence;
                viscosityForce.second += (particles[t].dy - particles[sampleIndex].dy) * infuence;
        }
    }
    return viscosityForce;
}

FloatPair eachPointWithinRadius(int sampleIndex){
    Particle sampleParticle = particles[sampleIndex];

    FloatPair cell = posToCellCoord(sampleParticle, SMOOTHRADIUS);
    float centerX = cell.first;
    float centerY = cell.second;

    float sqRadius = SMOOTHRADIUS * SMOOTHRADIUS;

    FloatPair pressureForce;
    pressureForce.first = 0;
    pressureForce.second = 0;

    for(int i=0;i<9; i++){
        int key = getKeyFromHash(hashCell(centerX + cellOffsets[i][0], centerY + cellOffsets[i][1]));
        
        int cellStartIndex = startIndex[key];
        
        int sizeOfSL = sizeof(spatialLookup) / sizeof(spatialLookup[0]);

        for(int t=cellStartIndex; t<sizeOfSL; t++){

            if(spatialLookup[t] != key) break;

            int particleIndex = t;

            float dist_X = particles[particleIndex].pred_x - sampleParticle.pred_x;
            float dist_Y = particles[particleIndex].pred_y - sampleParticle.pred_y;
            float sqDist = (dist_X*dist_X + dist_Y*dist_Y);

            if(sqDist <= sqRadius){
                if(particleIndex == sampleIndex) continue;
                
                float dist = sqrt(sqDist);

                float dir_X = (dist == 0) ? GetRandomDir() : dist_X / dist;
                float dir_Y = (dist == 0) ? GetRandomDir() : dist_Y / dist;

                float slope = smoothingKernalDerivative_new(SMOOTHRADIUS, dist);
                float density = densities[particleIndex];
                float sharedPressure = calculateSharedPressure(density, densities[sampleIndex]);
                pressureForce.first += sharedPressure * dir_X * slope * MASS / density;
                pressureForce.second -= sharedPressure * dir_Y * slope * MASS / density;
            } 
        }
    }
    return pressureForce;
}

FloatPair calculatePressureForce(int particleIndex){
    return eachPointWithinRadius(particleIndex);
}

void calculateDensityForces(){
    for(int i=0; i<NUM_PARTICLES; i++){
        FloatPair pressureForce = calculatePressureForce(i);
        float pressureAccel_X = pressureForce.first / densities[i];
        float pressureAccel_Y = pressureForce.second / densities[i];
        FloatPair viscosityForce = calculateViscosityForce(i);
        pressureAccel_X += viscosityForce.first * VISCOSITY;
        pressureAccel_Y += viscosityForce.second * VISCOSITY;

        particles[i].dx += pressureAccel_X * DT;
        particles[i].dy += pressureAccel_Y * DT;

    }
}

float calculateDensity(Particle samplePoint){
    float density = 0;

    for(int i=0; i<NUM_PARTICLES; i++){
        float dist_x = samplePoint.pred_x - particles[i].pred_x;
        float dist_y = samplePoint.pred_y - particles[i].pred_y;

        float dist = sqrt(dist_x * dist_x + dist_y * dist_y);
        float influence = smoothingKernal_new(SMOOTHRADIUS, dist);
        density += MASS * influence;
    }

    return density;
}

void calculateDensities(){
    float avg = 0;
    for(int i=0;i<NUM_PARTICLES; i++){
        densities[i] = calculateDensity(particles[i]);
        avg += densities[i];
    }
    printf("Avg Density %f \n", avg / NUM_PARTICLES);
}

void applyGravity(int i){
    particles[i].dy -= 1;
}

void resolveCollisions(int i){
    if(particles[i].x > L){
        particles[i].x = L - 0.1;
        particles[i].dx *= -DECAY;
    } else if(particles[i].x < 50) {
        particles[i].x = 50 + 0.1;
        particles[i].dx *= -DECAY;
    }
    if(particles[i].y > L){
        particles[i].y = L - 0.1;
        particles[i].dy *= -DECAY;
    } else if(particles[i].y < 50){
        particles[i].y = 50 + 0.1;
        particles[i].dy *= -DECAY;
    }
}

void updateSpatialLookup(){
    for(int i=0; i<NUM_PARTICLES; i++){
        FloatPair cell = posToCellCoord(particles[i], SMOOTHRADIUS);
        int cellKey = getKeyFromHash(hashCell(cell.first, cell.second));
        spatialLookup[i] = cellKey;

        startIndex[i] = -99;
    }

    int sizeOfSL = sizeof(spatialLookup) / sizeof(spatialLookup[0]);
    qsort(spatialLookup, sizeOfSL, sizeof(float), compare);

    for(int t=0; t<NUM_PARTICLES; t++){
        int key = spatialLookup[t];
        int keyPrev = (t == 0) ? -99 : spatialLookup[t-1];

        if(key != keyPrev){
            startIndex[key] = t;
        }
    }
}

void updateParticles() {

    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].pred_x = particles[i].x + particles[i].dx * DT;
        particles[i].pred_y = particles[i].y - particles[i].dy * DT;
    }

    updateSpatialLookup();

    calculateDensities();
    calculateDensityForces();

    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].x += particles[i].dx * DT;
        particles[i].y -= particles[i].dy * DT;

        //printf("Pos: %f %f \n", particles[i].x, particles[i].y);

        //printf("Particles: %f, %f \n", particles[i].x, particles[i].y);
        //applyGravity(i);
        resolveCollisions(i);
    }
}

void renderParticles(SDL_Renderer *renderer) {
    //SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White color
    for (int i = 0; i < NUM_PARTICLES; i++) {
        // Calculate speed magnitude
        float speed = sqrt(particles[i].dx * particles[i].dx + particles[i].dy * particles[i].dy);
        //printf("Speed %f \n", speed);
        // Map speed to a color gradient
        Uint8 r, g, b;
        if (speed < 10.0) {
            // Slow particles are blue
            r = 0;
            g = 0;
            b = 255;
        } else if (speed < 100.0) {
            // Medium-speed particles are green
            r = 0;
            g = 255;
            b = 0;
        } else {
            // Fast particles are red
            r = 255;
            g = 0;
            b = 0;
        }

        // Set particle color
        SDL_SetRenderDrawColor(renderer, r, g, b, 255);
        SDL_RenderDrawPoint(renderer, (int)particles[i].x, (int)particles[i].y);
    }
}

int main(){

    srand(time(NULL));


    SDL_Window *window;
    SDL_Renderer *renderer;

    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow("Fluid Sim", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);

    // Initialize particle positions and velocities
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].x = (rand() % 100) + WINDOW_WIDTH /2;
        particles[i].y = (rand() % 100) + WINDOW_HEIGHT/2;
        particles[i].dx = 0; 
        particles[i].dy = 0; 
    }

    SDL_Event event;
    int quit = 0;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                quit = 1;
        }

        Uint32 currentTime = SDL_GetTicks();
        calculateFPS(currentTime);

        // Print FPS to terminal
        //printf("\rFPS: %.2f", fps);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black background
        SDL_RenderClear(renderer);

        updateParticles();
        renderParticles(renderer);

        SDL_RenderPresent(renderer);

        //sleep(1);
        //fflush(stdout);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}