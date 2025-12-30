import argparse
import random
import os
from datetime import datetime
from types import SimpleNamespace
import csv
import itertools

from train_loop import (
    load_yaml_config,
    merge_config_with_cli,
    select_gnn_features,
    setup_and_train_classifier,
    setup_and_train,
)


# --------------------------------------------------------
# 1. Feature configuration files
# --------------------------------------------------------
FEATURE_CONFIG_FILES = [
#    "data/ml_models/hyper_parameter/features/nn/edge_basic.yaml",
#    "data/ml_models/hyper_parameter/features/nn/edge_full.yaml",
#    "data/ml_models/hyper_parameter/features/nn/edge_light.yaml",
    "data/ml_models/hyper_parameter/features/nn/edge_no_rw_no_edgeeval.yaml",
]

# --------------------------------------------------------
# 2. Architectures
# --------------------------------------------------------
#ARCHITECTURES = ["GAT","GAT3", "GCN", "GCN3"]
ARCHITECTURES = [ "GCNStats", "GCN3Stats", "MLP"]

#ARCHITECTURES = ["GATStats","GAT3Stats", "GCNStats", "GCN3Stats", "MLP"]
#ARCHITECTURES = ["GATStats"]
#ARCHITECTURES = ["MLP4"]
#ARCHITECTURES = ["GAT"]
#ARCHITECTURES = [ "GCN"]
# --------------------------------------------------------
# 3. Hyperparameter search ranges
# --------------------------------------------------------
SEARCH_SPACE = {
    "hidden_channels": [32, 64, 128],
    #"hidden_channels": [ 64, 128],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "epochs": [ 300],
   "heads": [8],               # used for GAT
  # "heads":8,
    "batch_size": [64, 128, 256],
    "pooling": ['mean'] #Used for regression
  #  "pooling": ['mean','sum'] #Used for regression
}
# SEARCH_SPACE = {
#     "hidden_channels": [32, ],
#     "learning_rate": [3e-3],
#     "epochs": [ 1],
#     "heads": [8, ],               # used for GAT
#     "batch_size": [64, ],
# }

# --------------------------------------------------------
# Helper
# --------------------------------------------------------
def sample_param(options):
    return random.choice(options)


def save_trial_results(csv_path, trial_data, write_header=False):
    """
    Append trial results to CSV file.
    
    Args:
        csv_path: Path to the CSV file
        trial_data: Dictionary containing trial information
        write_header: Whether to write the header row
    """
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trial_data.keys())
        
        # Write header if file is new or write_header is True
        if not file_exists or write_header:
            writer.writeheader()
        
        writer.writerow(trial_data)
    
    print(f"💾 Results saved to: {csv_path}")


def unified_search(
    search_type,
    model_type,
    output_dir,
    search_space=SEARCH_SPACE,
    n_trials=None,
    data_seed=None
):
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = f"{search_type}_search_results_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # Multi-valued heads support
    heads_list = (
        search_space["heads"] if isinstance(search_space["heads"], list)
        else [search_space["heads"]]
    )

    # Hyperparameter parameter names and values
    param_names = [
        "hidden_channels",
        "learning_rate",
        "epochs",
        "heads",
        "batch_size",
        "pooling"
    ]
    param_values = [
        search_space["hidden_channels"],
        search_space["learning_rate"],
        search_space["epochs"],
        heads_list,
        search_space["batch_size"],
        search_space["pooling"]
    ]

    # ------------------------------------------------------------------
    # 1. Create trial iterator
    # ------------------------------------------------------------------
    if search_type == "grid":

        # FULL GRID = features × architectures × hyperparameters
        grid = list(
            itertools.product(
                FEATURE_CONFIG_FILES,
                ARCHITECTURES,
                *param_values
            )
        )

        # trial index and trial params
        trial_iterator = enumerate(grid, start=1)
        total_trials = len(grid)

    elif search_type == "random":
        if n_trials is None:
            raise ValueError("n_trials must be provided for random search.")

        def random_sample_generator():
            for i in range(1, n_trials + 1):
                yield i, (
                    random.choice(FEATURE_CONFIG_FILES),
                    random.choice(ARCHITECTURES),
                    *[random.choice(param_values[j]) for j in range(len(param_values))]
                )

        trial_iterator = random_sample_generator()
        total_trials = n_trials

    else:
        raise ValueError("search_type must be 'random' or 'grid'.")

    print(f"Total scheduled trials: {total_trials}")

    # ------------------------------------------------------------------
    # 2. Execute trials
    # ------------------------------------------------------------------
    for idx, combo in trial_iterator:

        print("\n" + "=" * 60)
        print(f"🔍 {search_type.capitalize()} Search Trial {idx}/{total_trials}")
        print("=" * 60)

        # Unpack combination
        (
            features_fp,
            architecture,
            hidden_channels,
            learning_rate,
            epochs,
            heads,
            batch_size,
            pooling,
        ) = combo

        # -------------------------------------------------------------
        # Step A: Load feature config
        # -------------------------------------------------------------
        print(f"Selected features: {features_fp}")
        cfg_data = load_yaml_config(features_fp)

        if "features" not in cfg_data:
            raise ValueError(f"Config file {features_fp} must contain a `features:` list.")
        w_graph_data = False
        if model_type == 'graph_regression':
            w_graph_data = True
        x_features, edge_features, node_features, graph_features = select_gnn_features(cfg_data["features"], w_graph_data)

        # -------------------------------------------------------------
        # Step B: Build HP namespace
        # -------------------------------------------------------------
        sampled_hp = {
            "hidden_channels": hidden_channels,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "heads": heads,
            "batch_size": batch_size,
            "pooling": pooling,
            "architecture": architecture,
            "model_type": model_type,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_output = os.path.join(output_dir, f"trial_{idx}_{timestamp}")
        os.makedirs(trial_output, exist_ok=True)
        sampled_hp["checkpoint_dir"] = trial_output

        print("🔧 Hyperparameters:")
        for k, v in sampled_hp.items():
            print(f"  {k}: {v}")

        cfg = SimpleNamespace(**sampled_hp)

        # -------------------------------------------------------------
        # Step C: Training call
        # -------------------------------------------------------------
        print(f"📁 Saving outputs to: {trial_output}")

        try:
            if cfg.model_type == "edge_classification":
                losses_dict = setup_and_train_classifier(
                    hidden_channels=cfg.hidden_channels,
                    learning_rate=cfg.learning_rate,
                    epochs=cfg.epochs,
                    heads=cfg.heads,
                    batch_size=cfg.batch_size,
                    model=cfg.architecture,
                    checkpoint_dir=cfg.checkpoint_dir,
                    x_features=x_features,
                    node_features=node_features,
                    edge_features=edge_features,
                    graph_features=graph_features,
                    seed=data_seed,
                )

            elif cfg.model_type == "graph_regression":
                losses_dict = setup_and_train(
                    hidden_channels=cfg.hidden_channels,
                    learning_rate=cfg.learning_rate,
                    epochs=cfg.epochs,
                    heads=cfg.heads,
                    batch_size=cfg.batch_size,
                    model=cfg.architecture,
                    checkpoint_dir=cfg.checkpoint_dir,
                    x_features=x_features,
                    node_features=node_features,
                    edge_features=edge_features,
                    graph_features=graph_features,
                    seed=data_seed,
                    pooling = cfg.pooling,
                )

            best_train_loss = min(losses_dict["train_losses"]) if losses_dict["train_losses"] else None
            best_test_loss = min(losses_dict["test_losses"]) if losses_dict["test_losses"] else None
            final_train_loss = losses_dict["train_losses"][-1] if losses_dict["train_losses"] else None
            final_test_loss = losses_dict["test_losses"][-1] if losses_dict["test_losses"] else None
            status = "SUCCESS"
            error_message = ""

        except Exception as e:
            print(f"❌ Trial {idx} failed: {str(e)}")
            best_train_loss = None
            best_test_loss = None
            final_train_loss = None
            final_test_loss = None
            status = "FAILED"
            error_message = str(e)

        # -------------------------------------------------------------
        # Step D: Write CSV entry
        # -------------------------------------------------------------
        row = {
            "trial": idx,
            "timestamp": timestamp,
            "model_type": cfg.model_type,
            "architecture": cfg.architecture,
            "hidden_channels": cfg.hidden_channels,
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.epochs,
            "heads": cfg.heads,
            "batch_size": cfg.batch_size,
            "pooling": cfg.pooling,
            "features_config": os.path.basename(features_fp),
            "best_train_loss": best_train_loss,
            "best_test_loss": best_test_loss,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "checkpoint_dir": trial_output,
            "status": status,
            "error": error_message,
        }

        save_trial_results(csv_path, row, write_header=(idx == 1))

        print(f"✅ Trial {idx} complete.")
        if status == "SUCCESS":
            print(f"   Best train loss: {best_train_loss}")
            print(f"   Best test loss: {best_test_loss}")

    print("\n" + "=" * 60)
    print(f"🎉 {search_type.capitalize()} search complete. Results written to {csv_path}")
    print("=" * 60)

# --------------------------------------------------------
# CLI
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Random Search Tuning Loop")

    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of random experiments")
    parser.add_argument("--data_seed", type=int, default=None,
                        help="Seed for data splitting (keep consitency accross models)")
    parser.add_argument("--model_type", type=str, default="edge_classification",
                        choices=["edge_classification", "graph_regression"],
                        help="What to train")
    parser.add_argument('--search_type',type=str, default="random",
                        choices=["random", "grid"],
                        help="Grid search or random sample of grid")

    parser.add_argument("--output_dir", type=str, default="tuning_runs",
                        help="Directory to store trial results")

    args = parser.parse_args()

    unified_search(
        search_type=args.search_type,
        n_trials=args.n_trials,
        model_type=args.model_type,
        output_dir=args.output_dir,
        data_seed=args.data_seed
    )


if __name__ == "__main__":
    main()
