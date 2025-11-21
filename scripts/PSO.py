from numpy import array
import random
import numpy as np


class Particle:
    """
    Represents a single particle in the PSO swarm
    
    Each particle has:
    - A position (current solution in search space)
    - A velocity (direction and speed of movement)
    - Personal best position and fitness
    - A set of informants (itself + neighbors)
    """
    
    def __init__(self, num_dimensions, lower_bound=-10, upper_bound=10):
        # Initialize position randomly within bounds (Line 9)
        self.position = array([random.uniform(lower_bound, upper_bound) for _ in range(num_dimensions)]) 
        # Initialize velocity randomly (Line 9)
        velocity_range = abs(upper_bound - lower_bound)
        self.velocity = array([random.uniform(-velocity_range, velocity_range) for _ in range(num_dimensions)])
        self.pBest_position = self.position.copy()
        self.pFitness = float('inf')  
        self.informants = []
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    # Apply boundary constraints to the particle's position
    def apply_bounds(self):
        self.position = np.clip(self.position, self.lower_bound, self.upper_bound)


class PSO:
    """
    Particle Swarm Optimization algorithm
    
    Optimizes parameters by a swarm of particles.
    Each particle represents a candidate set of weights and biases.
    Args:
            swarm_size: Number of particles in the swarm
            alpha: Inertia weight
            beta: Cognitive weight
            gamma: Social weight
            delta: Global weight
            epsilon: Step size
            num_dimensions: Number of dimensions in search space
            max_iter: Maximum number of iterations
            num_informants: Number of informants per particle 
    """
    #Line 1-6 of psuedocode
    def __init__(self, swarm_size, alpha, beta, gamma, delta, epsilon, 
                 num_dimensions, max_iter, num_informants):
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.alpha = alpha  
        self.beta = beta  
        self.gamma = gamma  
        self.delta = delta  
        self.epsilon = epsilon  
        self.gBest_position = None     
        self.gBest_fitness = float('inf')  
        self.num_informants = num_informants
        self.num_dimensions = num_dimensions
        self.particles = [Particle(num_dimensions) for _ in range(swarm_size)]# Initialize swarm (Line 7-9)
        self.get_informants(num_informants)

    #Set up informant topology for each particle
    def get_informants(self, num_informants):
        for particle in self.particles:
            # Randomly select other particles as informants
            other_particles = [p for p in self.particles if p is not particle]
            informants = random.sample(other_particles, num_informants - 1)
            
            # Include the particle itself in its informant list
            particle.informants = [particle] + informants
    
    #Find the best position among a particle's informants
    def update_informants_best(self, particle):
        best_informant_fitness = float('inf')
        best_informant_position = particle.pBest_position 

        for informant in particle.informants:
            if informant.pFitness < best_informant_fitness:
                best_informant_fitness = informant.pFitness
                best_informant_position = informant.pBest_position

        return best_informant_position
    
    #Evaluate particle's fitness and update best positions(Line 12-15 of psuedocode)
    def update_fitness(self, particle, ann, X_train, Y_train):
        # Set parameters from particle's position
        ann.set_weights_bias(particle.position)
        
        # Get predictions on training data
        y_pred = ann.feedforward(X_train)
        
        # Calculate fitness with mean absolute error
        fitness = np.mean(np.abs(Y_train - y_pred))

        # Update personal best (pBest) if current position is better (Line 14)
        if fitness < particle.pFitness: 
            particle.pBest_position = particle.position.copy() # Line 15
            particle.pFitness = fitness
        
        # Update global best (gBest) if current position is better
        if fitness < self.gBest_fitness:
            self.gBest_position = particle.position.copy()
            self.gBest_fitness = fitness

        return fitness
    
    #Update particle velocity using PSO velocity equation (Line 16-24 of psuedocode)
    def velocity_function(self, particle, informantBest):
        randBeta = random.uniform(0.0, self.beta) # Line 21
        randGamma = random.uniform(0.0, self.gamma)# Line 22
        randDelta = random.uniform(0.0, self.delta)# Line 23

        # Update velocity: inertia + personal best + informant best + global best (Line 21-24)
        particle.velocity = (self.alpha * particle.velocity +  
                            randBeta * (particle.pBest_position - particle.position) +  
                            randGamma * (informantBest - particle.position)) 
             
        if self.gBest_position is not None:
            particle.velocity += randDelta * (self.gBest_position - particle.position)
        
    #Main PSO loop to move particles and optimize parameters (Line 8-28 of psuedocode)             
    def move_particle(self, ann, X_train, Y_train):
        for iter in range(self.max_iter):

            # Process each particle in the swarm
            for particle in self.particles: 
                self.update_fitness(particle, ann, X_train, Y_train) # (Line 12-15)

                informantBest = self.update_informants_best(particle)
                
                self.velocity_function(particle, informantBest)# (Line 16-24)
                
                particle.position += self.epsilon * particle.velocity # Line 26
                
                # Apply boundary constraints
                particle.apply_bounds()
            
            print(f"Iteration {iter + 1}, Global Best Fitness: {self.gBest_fitness}")

        # Return the best solution found
        return self.gBest_position, self.gBest_fitness# Line 28






       

   

