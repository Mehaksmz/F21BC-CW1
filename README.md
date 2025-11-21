# F21BC-CW1 – ANN + PSO Concrete Strength Modelling

This project trains feedforward Artificial Neural Networks (ANNs) to predict concrete compressive strength. Particle Swarm Optimization (PSO) is used to tune network weights, biases, and activation functions. The repository contains scripts for training a single model, running hyper-parameter sweeps, and plotting the resulting metrics.

## Repository Layout

```
.
├── data/
│   └── concrete_data.csv        # UCI concrete strength dataset (features + target)
├── scripts/
│   ├── ANN.py                   # Feedforward ANN implementation
│   ├── PSO.py                   # Particle Swarm Optimiser with informant topology
│   ├── regression.py            # Use dataset to train ANN with PSO
│   ├── hyperparameter_test.py   # Experiment parameters for ANN/PSO 
│   └── results_plot.py          # Visualisations for experiment_results.csv
├── experiment_results.csv       # Saved metrics from hyperparameter_test.py
└── README.md
```

- `data/`: Raw dataset (expects the CSV to stay in this folder).
- `scripts/ANN.py`: Handles network initialisation, feedforward passes, and parameter reshaping for PSO.
- `scripts/PSO.py`: Implements particles, velocity updates, and the optimisation loop. Inline comments map directly to pseudocode line numbers from the report.
- `scripts/regression.py`: Loads data, runs ANN with PSO, and reports MAE/MSE/RMSE on the test split.
- `scripts/hyperparameter_test.py`: Automates multiple runs over different ANN architectures, activation functions, and PSO hyper-parameters. Outputs summary statistics to `experiment_results.csv`.
- `scripts/results_plot.py`: Reads the CSV, aggregates MAE by architecture, and creates bar/heatmap plots for quick comparison.

## Requirements

- Python 3.10+
- Recommended packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Install dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

1. **Prepare data**  
   Ensure `data/concrete_data.csv` is present. 

2. **Run single training/evaluation**  
   Trains one ANN (default `[input, 10, output]`) and optimises it via PSO.
   ```bash
   cd scripts
   python regression.py
   ```

3. **Run hyper-parameter experiments**  
   Loops over the configurations defined in `hyperparameter_test.py`, performs multiple stochastic runs, and saves aggregate metrics.
   ```bash
   cd scripts
   python hyperparameter_test.py
   ```
   Results are written to `experiment_results.csv` in the project root.

4. **Plot results**  
   Generate bar charts and heatmaps using the saved CSV.
   ```bash
   cd scripts
   python results_plot.py
   ```

