import numpy as np
import pandas as pd
from ANN import ANN
from PSO import PSO
from regression import load_concrete_data

num_runs = 10  # independent runs per config

ann_configs = [
    [8, 5, 1],      # 1 hidden layer, 5 neurons
    [8, 10, 1],     # 1 hidden layer, 10 neurons
    [8, 10, 5, 1],  # 2 hidden layers
]

activation_configs = [
    ["relu", "sigmoid"],
    ["tanh", "sigmoid"],
    ["relu", "linear", "sigmoid"]
]

pso_params_list = [
    {"swarm_size": 20, "max_iter": 30, "alpha": 0.5, "beta": 1.8, "gamma": 1.8, "delta":0.4, "epsilon":0.8, "num_informants": 3},
    {"swarm_size": 30, "max_iter": 50, "alpha": 0.5, "beta": 1.8, "gamma": 1.8, "delta":0.4, "epsilon":0.8, "num_informants": 5},
]

X_train, X_test, y_train, y_test = load_concrete_data("data/concrete_data.csv")

results = []

for ann_nodes, act_funcs in zip(ann_configs, activation_configs):
    num_layers = len(ann_nodes)
    for pso_params in pso_params_list:
        maes, mses, rmses = [], [], []

        print(f"\n=== ANN: {ann_nodes} | Activations: {act_funcs} | PSO: {pso_params} ===\n")

        for run in range(num_runs):
            ann = ANN(num_layers, ann_nodes, act_funcs)

            weight_count = sum(ann_nodes[i] * ann_nodes[i+1] for i in range(len(ann_nodes)-1))
            bias_count = sum(ann_nodes[i+1] for i in range(len(ann_nodes)-1))
            num_functions = len(ann_nodes) - 1
            dimensions = weight_count + bias_count + num_functions

            pso = PSO(
                swarm_size=pso_params["swarm_size"],
                alpha=pso_params["alpha"],
                beta=pso_params["beta"],
                gamma=pso_params["gamma"],
                delta=pso_params["delta"],
                epsilon=pso_params["epsilon"],
                num_dimensions=dimensions,
                max_iter=pso_params["max_iter"],
                num_informants=pso_params["num_informants"]
            )

            best_position, best_fitness = pso.move_particle(ann, X_train, y_train)
            ann.set_weights_bias(best_position)

            y_test_pred = ann.feedforward(X_test)
            mae = np.average(np.abs(y_test - y_test_pred))
            mse = np.average((y_test - y_test_pred)**2)
            rmse = np.sqrt(mse)

            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)

            print(f"Run {run+1}/{num_runs} => MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        # Store mean and std
        results.append({
            "ANN_Nodes": ann_nodes,
            "Activations": [f.__name__ for f in ann.functions],
            "PSO_Params": pso_params,
            "MAE_avg": np.average(maes),
            "MAE_std": np.std(maes),
            "MSE_avg": np.average(mses),
            "MSE_std": np.std(mses),
            "RMSE_avg": np.average(rmses),
            "RMSE_std": np.std(rmses)
        })

df_results = pd.DataFrame(results)
df_results.to_csv("experiment_results.csv", index=False)
print("\nAll experiments completed. Results saved to 'experiment_results.csv'.")