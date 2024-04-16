
#include <SDL2/SDL.h>  

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define NUM_PARTICLES 100

typedef struct {
    float x;
    float y; 
    float dx;
    float dy;
} Particle;

Particle particles[NUM_PARTICLES];

void updateParticles() {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].x += particles[i].dx;
        particles[i].y += particles[i].dy;
    }
}

void renderParticles(SDL_Renderer *renderer) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White color
    for (int i = 0; i < NUM_PARTICLES; i++) {
        SDL_RenderDrawPoint(renderer, (int)particles[i].x, (int)particles[i].y);
    }
}

int main(){
    SDL_Window *window;
    SDL_Renderer *renderer;

    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow("Fluid Sim", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);

    // Initialize particle positions and velocities
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].x = rand() % WINDOW_WIDTH;
        particles[i].y = rand() % WINDOW_HEIGHT;
        particles[i].dx = ((float)rand() / RAND_MAX - 0.5) * 2; // Random velocity in x direction
        particles[i].dy = ((float)rand() / RAND_MAX - 0.5) * 2; // Random velocity in y direction
    }

    SDL_Event event;
    int quit = 0;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                quit = 1;
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black background
        SDL_RenderClear(renderer);

        updateParticles();
        renderParticles(renderer);

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}