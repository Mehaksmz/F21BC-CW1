import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Import your ANN and PSO classes
from ANN import ANN
from PSO import PSO, Particle

class Regression:
    def _init_(self):
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.ann = None
        self.best_params = None
        
    def load_data(self):
        """Load the concrete compressive strength dataset"""
        print("Loading dataset...")
        concrete_data = fetch_ucirepo(id=165)
        
        X = concrete_data.data.features.values
        y = concrete_data.data.targets.values
        
        # Split into train and test sets (70-30 split)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        
        # Normalize the data
        self.X_train = self.scaler_X.fit_transform(self.X_train)
        self.X_test = self.scaler_X.transform(self.X_test)
        
        self.Y_train = self.scaler_Y.fit_transform(self.Y_train)
        self.Y_test = self.scaler_Y.transform(self.Y_test)
        
        print("Data loaded and normalized successfully!")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"Y_train shape: {self.Y_train.shape}")
        
    def create_ann(self, hidden_layers=[10, 5], activation='relu'):
        """Create ANN architecture"""
        # Input layer has 8 nodes (8 features), output layer has 1 node
        nodes = [8] + hidden_layers + [1]
        num_layers = len(nodes)
        
        # Functions for each layer transition
        # Use activation for hidden layers, but we need linear output for regression
        # Since your ANN doesn't have 'linear', we'll use tanh for output (closer to linear than relu)
        functions = [activation] * (num_layers - 2) + ['tanh']
        
        self.ann = ANN(num_layers, nodes, functions)
        print(f"ANN created with architecture: {nodes}")
        print(f"Activations: {functions}")
        
    def calculate_num_params(self):
        """Calculate total number of parameters (weights + biases)"""
        total_params = 0
        for i in range(len(self.ann.ws)):
            total_params += np.prod(self.ann.ws[i].shape)  # weights
            total_params += np.prod(self.ann.bs[i].shape)  # biases
        return total_params
    
    def objective_function_batch(self, params):
        """Objective function that evaluates MAE on entire training set"""
        self.ann.set_weights_bias(params)
        
        # Predict on all training samples
        predictions = []
        for i in range(len(self.X_train)):
            pred = self.ann.feedforward(self.X_train[i])
            predictions.append(pred[0, 0])
        
        predictions = np.array(predictions).reshape(-1, 1)
        mae = np.mean(np.abs(self.Y_train - predictions))
        return mae
    
    def train(self, swarm_size=30, max_iter=50, num_informants=3):
        """Train the ANN using YOUR PSO implementation"""
        if self.ann is None:
            print("Creating default ANN architecture...")
            self.create_ann()
        
        num_dimensions = self.calculate_num_params()
        print(f"Total parameters to optimize: {num_dimensions}")
        
        # PSO hyperparameters - tuned for better exploration
        alpha = 0.5   # velocity retention (lower = more exploration)
        beta = 1.8    # personal best influence
        gamma = 1.8   # informants best influence
        delta = 0.4   # global best influence (increased)
        epsilon = 0.8 # step size (reduced for stability)
        
        print("\nStarting PSO training...")
        print("="*60)
        
        # Create PSO instance using YOUR class
        pso = PSO(
            swarm_size=swarm_size,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            num_dimensions=num_dimensions,
            obj_func=self.objective_function_batch,
            max_iter=max_iter,
            num_informants=num_informants
        )
        
        # Training loop (fixing your PSO logic)
        for iteration in range(max_iter):
            for particle in pso.particles:
                # Evaluate fitness
                fitness = self.objective_function_batch(particle.position)
                
                # Update personal best
                if fitness < particle.pFitness:
                    particle.pBest_position = particle.position.copy()
                    particle.pFitness = fitness
                
                # Update global best
                if fitness < pso.gBest_fitness:
                    pso.gBest_position = particle.position.copy()
                    pso.gBest_fitness = fitness
                
                # Get informant best
                best_informant_fitness = float('inf')
                best_informant_position = particle.pBest_position
                
                for informant in particle.informants:
                    if informant.pFitness < best_informant_fitness:
                        best_informant_fitness = informant.pFitness
                        best_informant_position = informant.pBest_position
                
                # Update velocity using YOUR velocity function logic
                rand_beta = np.random.uniform(0, pso.beta)
                rand_gamma = np.random.uniform(0, pso.gamma)
                rand_delta = np.random.uniform(0, pso.delta)
                
                particle.velocity = (pso.alpha * particle.velocity +
                                   rand_beta * (particle.pBest_position - particle.position) +
                                   rand_gamma * (best_informant_position - particle.position) +
                                   rand_delta * (pso.gBest_position - particle.position))
                
                # Update position
                particle.position += pso.epsilon * particle.velocity
                particle.apply_bounds()
            
            # Print progress
            if iteration % 5 == 0 or iteration == max_iter - 1:
                print(f"Iteration {iteration + 1}/{max_iter} | Best MAE: {pso.gBest_fitness:.6f}")
        
        print("="*60)
        self.best_params = pso.gBest_position
        self.ann.set_weights_bias(self.best_params)
        print(f"Training complete! Final training MAE: {pso.gBest_fitness:.6f}")
        
    def evaluate(self):
        """Evaluate the trained ANN on test set"""
        if self.best_params is None:
            print("Model not trained yet!")
            return None
        
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        predictions = []
        for i in range(len(self.X_test)):
            pred = self.ann.feedforward(self.X_test[i])
            predictions.append(pred[0, 0])
        
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Calculate MAE on normalized data
        mae_normalized = np.mean(np.abs(self.Y_test - predictions))
        
        # Denormalize for actual MAE in MPa
        y_test_actual = self.scaler_Y.inverse_transform(self.Y_test)
        predictions_actual = self.scaler_Y.inverse_transform(predictions)
        mae_actual = np.mean(np.abs(y_test_actual - predictions_actual))
        
        # Additional metrics
        mse_actual = np.mean((y_test_actual - predictions_actual)**2)
        rmse_actual = np.sqrt(mse_actual)
        
        # Calculate max and min errors to see the range
        errors = np.abs(y_test_actual - predictions_actual)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        print(f"\nTest Results:")
        print(f"  MAE (normalized): {mae_normalized:.6f}")
        print(f"  MAE (actual):     {mae_actual:.4f} MPa")
        print(f"  RMSE (actual):    {rmse_actual:.4f} MPa")
        print(f"  Max Error:        {max_error:.4f} MPa")
        print(f"  Min Error:        {min_error:.4f} MPa")
        print(f"\nSample predictions (first 10):")
        for i in range(min(10, len(y_test_actual))):
            error = abs(y_test_actual[i,0] - predictions_actual[i,0])
            print(f"  Expected: {y_test_actual[i,0]:6.2f} MPa | Predicted: {predictions_actual[i,0]:6.2f} MPa | Error: {error:5.2f}")
        print("="*60)
        
        return mae_actual


# Main execution
if _name_ == "_main_":
    # Create regression object
    reg = Regression()
    
    # Load and preprocess data
    reg.load_data()
    
    # Create ANN with custom architecture
    # Larger network for more learning capacity
    reg.create_ann(hidden_layers=[16, 8], activation='relu')
    
    # Train the ANN using PSO with better exploration
    reg.train(swarm_size=50, max_iter=150, num_informants=5)
    
    # Evaluate on test set
    mae = reg.evaluate()
    
    print(f"\n🎯 Final Test MAE: {mae:.4f} MPa")