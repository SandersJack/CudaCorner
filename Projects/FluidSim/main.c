#include "CudaKernal.h"

#include <SDL2/SDL.h>  
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h> 
#include <time.h> 
#include <math.h>

#define M_PI 3.14159265358979323846

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define NUM_PARTICLES 100000

int L = WINDOW_WIDTH - 50;

#define DECAY 0.5
#define MASS 1
#define SMOOTHRADIUS 100
#define TARGET_DENSITY 0.002 //0.021
#define PRESSURE_MULT 50
#define VISCOSITY 10

#define DT 0.05

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


void renderParticles(SDL_Renderer *renderer, Particle *particles) {
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

    int* num_particles = (int*)malloc(sizeof(int));
    *num_particles = NUM_PARTICLES;

    float* dt = (float*)malloc(sizeof(float));
    *dt = DT;

    Particle* particles = (Particle*)malloc(*num_particles * sizeof(Particle));

    float* densities = (float*)malloc(*num_particles * sizeof(float));

    int* spatialLookup = (int*)malloc(*num_particles * sizeof(int));
    SpacialIndex* spacialIndexs = (SpacialIndex*)malloc(*num_particles * sizeof(SpacialIndex));

    FloatPair* pressureForces = (FloatPair*)malloc(*num_particles * sizeof(FloatPair));


    IntPair* spacialOffsets = (IntPair*)malloc(9 * sizeof(IntPair));

    int offsets[9][2] = {
        {-1, 1},
        {0, 1},
        {1, 1},
        {-1, 0},
        {0, 0},
        {1, 0},
        {-1, -1},
        {0, -1},
        {1, -1}
    };

    for(int i=0; i<9; i++){
        spacialOffsets[i].first = offsets[i][0]; spacialOffsets[i].second = offsets[i][1];
    }


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
        particles[i].pred_x = particles[i].x; 
        particles[i].pred_x = particles[i].y; 
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

        printf("\rFPS: %.2f", fps);
        fflush(stdout);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black background
        SDL_RenderClear(renderer);


        __updateParticle(particles, dt, num_particles, densities, spatialLookup, spacialIndexs, pressureForces, spacialOffsets);

        renderParticles(renderer, particles);

        SDL_RenderPresent(renderer);
    }

    free(num_particles);
    free(dt);
    free(particles);
    free(densities);
    free(spatialLookup);
    free(spacialIndexs);
    free(pressureForces);
    free(spacialOffsets);



    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    cuda_kernel();

    return 0;
}