from numpy import array
import random
import numpy as np

class Particle:
    def __init__(self, num_dimensions, lower_bound=-10, upper_bound=10):
        # self.position = array([random.random() for _ in range(num_dimensions)])
        # self.velocity = array([random.random() for _ in range(num_dimensions)])
        self.position = array([random.uniform(lower_bound, upper_bound) for _ in range(num_dimensions)])
        velocity_range = abs(upper_bound - lower_bound)
        self.velocity = array([random.uniform(-velocity_range, velocity_range) for _ in range(num_dimensions)])
        self.pBest_position = self.position.copy()
        self.pFitness = float('inf')
        self.informants = []
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def apply_bounds(self):
        self.position = np.clip(self.position, self.lower_bound, self.upper_bound)

class PSO:
    def __init__(self, swarm_size, alpha, beta, gamma, delta, epsilon, num_dimensions, obj_func, max_iter, num_informants):
        self.swarm_size = swarm_size
        self.obj_func = obj_func
        self.max_iter = max_iter
        self.alpha = alpha #velocity retained
        self.beta = beta #personal best
        self.gamma = gamma #informants best
        self.delta = delta #global best
        self.epsilon = epsilon #jump size
        self.gBest_position = None
        self.gBest_fitness = float('inf')
        self.num_informants = num_informants
        self.num_dimensions = num_dimensions
        self.particles = [Particle(num_dimensions) for _ in range(swarm_size)]
        self.get_informants(num_informants)

    def get_informants(self, num_informants):
        for particle in self.particles:
          informants = random.sample([p for p in self.particles if p is not particle], num_informants - 1)
          particle.informants = [particle] + informants
          
    def update_informants_best(self, particle):
        best_informant_fitness = float('inf')
        best_informant_position = particle.pBest_position 

        for informant in particle.informants:
            if informant.pFitness < best_informant_fitness:
                best_informant_fitness = informant.pFitness
                best_informant_position = informant.pBest_position

        return best_informant_position
    
    def update_fitness(self, particle, ann, X_train, Y_train):
        ann.set_weights_bias(particle.position)
        y_pred = ann.feedforward(X_train)
        fitness = np.mean(np.abs(Y_train - y_pred))

        #Local best
        if fitness < particle.pFitness:
            particle.pBest_position = particle.position.copy()
            particle.pFitness = fitness
        #gBest
        if fitness < self.gBest_fitness:
            self.gBest_position = particle.position.copy()
            self.gBest_fitness = fitness

        return fitness

    def velocity_function(self, particle, informantBest):
        randBeta = random.uniform(0.0, self.beta)
        randGamma = random.uniform(0.0, self.gamma)
        randDelta = random.uniform(0.0, self.delta)

        particle.velocity = self.alpha * particle.velocity \
             + randBeta * (particle.pBest_position - particle.position) \
             + randGamma * (informantBest - particle.position) 
             
        if self.gBest_position is not None:
            particle.velocity += randDelta * (self.gBest_position - particle.position)
        
                   
    def move_particle(self, ann, X_train, Y_train):
        for iter in range(self.max_iter):
            for particle in self.particles:
                self.update_fitness(particle, ann, X_train, Y_train)
                informantBest = self.update_informants_best(particle)
                self.velocity_function(particle, informantBest)
                particle.position += self.epsilon * particle.velocity
                particle.apply_bounds()
            print(f"Iteration {iter + 1}, Global Best Fitness: {self.gBest_fitness}")

        return self.gBest_position, self.gBest_fitness






       

   

