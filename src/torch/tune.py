import argparse
import random
import os
from datetime import datetime
from types import SimpleNamespace
import csv


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
    "data/ml_models/hyper_parameter/features/nn/edge_basic.yaml",
    "data/ml_models/hyper_parameter/features/nn/edge_full.yaml",
    "data/ml_models/hyper_parameter/features/nn/edge_light.yaml",
    "data/ml_models/hyper_parameter/features/nn/edge_no_rw_no_edgeeval.yaml",
]

# --------------------------------------------------------
# 2. Architectures
# --------------------------------------------------------
ARCHITECTURES = ["GAT","GAT3", "GCN", "GCN3"]
#ARCHITECTURES = [ "GCN"]
# --------------------------------------------------------
# 3. Hyperparameter search ranges
# --------------------------------------------------------
SEARCH_SPACE = {
    "hidden_channels": [32, 64, 128],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "epochs": [ 200],
    "heads": [8, 16],               # used for GAT
    "batch_size": [64, 128, 256],
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


# --------------------------------------------------------
# 4. Tuning loop with custom output directory and CSV logging
# --------------------------------------------------------
def random_search(n_trials, model_type, output_dir):

    # ensure base dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create CSV file path
    csv_filename = f"random_search_results_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    for trial in range(n_trials):

        print("\n" + "=" * 60)
        print(f"🔍 Random Search Trial {trial+1}/{n_trials}")
        print("=" * 60)

        # -----------------------------
        # STEP A — Pick feature config
        # -----------------------------
        features_fp = random.choice(FEATURE_CONFIG_FILES)
        print(f"Selected features: {features_fp}")

        cfg_data = load_yaml_config(features_fp)

        if "features" not in cfg_data:
            raise ValueError(f"Config file {features_fp} must contain a `features:` list.")

        node_features, edge_features = select_gnn_features(cfg_data["features"])

        # ----------------------------------
        # STEP B — Randomly pick parameters
        # ----------------------------------
        sampled_hp = {
            "hidden_channels": sample_param(SEARCH_SPACE["hidden_channels"]),
            "learning_rate": sample_param(SEARCH_SPACE["learning_rate"]),
            "epochs": sample_param(SEARCH_SPACE["epochs"]),
            "heads": sample_param(SEARCH_SPACE["heads"]),
            "batch_size": sample_param(SEARCH_SPACE["batch_size"]),
            "architecture": random.choice(ARCHITECTURES),
            "model_type": model_type,
        }

        # ----------------------------------
        # STEP C — Create trial directory
        # ----------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_output = os.path.join(output_dir, f"trial_{trial+1}_{timestamp}")
        os.makedirs(trial_output, exist_ok=True)

        sampled_hp["checkpoint_dir"] = trial_output

        print("🔧 Sampled hyperparameters:")
        for k, v in sampled_hp.items():
            print(f"  {k}: {v}")

        cfg = SimpleNamespace(**sampled_hp)

        # ----------------------------------
        # STEP D — Run training
        # ----------------------------------
        print(f"📁 Saving outputs to: {trial_output}")
        
        try:
            # Losses dict contains the keys 'train_losses' and 'test_losses'
            if cfg.model_type == "edge_classification":
                losses_dict = setup_and_train_classifier(
                    hidden_channels=cfg.hidden_channels,
                    learning_rate=cfg.learning_rate,
                    epochs=cfg.epochs,
                    heads=cfg.heads,
                    batch_size=cfg.batch_size,
                    model=cfg.architecture,
                    checkpoint_dir=cfg.checkpoint_dir,
                    node_features=node_features,
                    edge_features=edge_features,
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
                    node_features=node_features,
                    edge_features=edge_features,
                )
            
            # ----------------------------------
            # STEP E — Extract best losses
            # ----------------------------------
            best_train_loss = min(losses_dict['train_losses']) if losses_dict['train_losses'] else None
            best_test_loss = min(losses_dict['test_losses']) if losses_dict['test_losses'] else None
            final_train_loss = losses_dict['train_losses'][-1] if losses_dict['train_losses'] else None
            final_test_loss = losses_dict['test_losses'][-1] if losses_dict['test_losses'] else None
            
            status = "SUCCESS"
            error_message = ""
            
        except Exception as e:
            print(f"❌ Trial {trial+1} failed with error: {str(e)}")
            best_train_loss = None
            best_test_loss = None
            final_train_loss = None
            final_test_loss = None
            status = "FAILED"
            error_message = str(e)
        
        # ----------------------------------
        # STEP F — Save results to CSV
        # ----------------------------------
        trial_data = {
            "trial": trial + 1,
            "timestamp": timestamp,
            "model_type": cfg.model_type,
            "architecture": cfg.architecture,
            "hidden_channels": cfg.hidden_channels,
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.epochs,
            "heads": cfg.heads,
            "batch_size": cfg.batch_size,
            "features_config": os.path.basename(features_fp),
            "best_train_loss": best_train_loss,
            "best_test_loss": best_test_loss,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "checkpoint_dir": trial_output,
            "status": status,
            "error": error_message,
        }
        
        save_trial_results(csv_path, trial_data, write_header=(trial == 0))
        
        print(f"✅ Trial {trial+1} complete.")
        if status == "SUCCESS":
            print(f"   Best train loss: {best_train_loss:.6f}")
            print(f"   Best test loss: {best_test_loss:.6f}")
    
    print("\n" + "=" * 60)
    print(f"🎉 Random search complete! Results saved to: {csv_path}")
    print("=" * 60)


# --------------------------------------------------------
# CLI
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Random Search Tuning Loop")

    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of random experiments")

    parser.add_argument("--model_type", type=str, default="edge_classification",
                        choices=["edge_classification", "graph_regression"],
                        help="What to train")

    parser.add_argument("--output_dir", type=str, default="tuning_runs",
                        help="Directory to store trial results")

    args = parser.parse_args()

    random_search(
        n_trials=args.n_trials,
        model_type=args.model_type,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
