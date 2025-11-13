#!.venv/bin/python3
import argparse
from pathlib import Path
from typing import Dict, Tuple 

from core.utils import (
    load_dataset,
    set_seed,
    create_data_loaders,
    count_parameters,
    save_model,
)

from core.dataset import RawDatasetList
from core.trainer import ModelTrainer
from core.evaluator import ModelEvaluator, plot_training_history
from models import SimpleMoleculeGCN, AdvancedMoleculeGNN
from config import ModelConfig, TrainingConfig


# contains map of { csv_file_path : (smiles_col, target_col)}
_dataset_column_map: Dict[RawDatasetList, Tuple[str, str]] = {
    RawDatasetList.ESOL :  ("smiles", "ESOL predicted log solubility in mols per litre"), 
    RawDatasetList.FREESOLV: ("smiles", "calc"), 
    RawDatasetList.LIPOPHILICITY: ("smiles", "exp")
}

def dataset_loader_from_map(dataset: RawDatasetList): 
    csv_path: str = "../data/raw/" + dataset.value
    smile_col = _dataset_column_map[dataset][0]
    target_col = _dataset_column_map[dataset][1]
    return load_dataset(
        csv_path, 
        smiles_col=smile_col, 
        target_col=target_col
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for flexible training runs."""
    parser = argparse.ArgumentParser(
        description="Train MoleculeGNN models with configurable datasets and architectures."
    )

    # Data options
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/ESOL.csv",
        help="Path to the dataset CSV file.",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="Name of the SMILES column in the dataset.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="measured log solubility in mols per litre",
        help="Name of the target column in the dataset.",
    )

    # Model options
    parser.add_argument(
        "--model",
        choices=["gcn", "gin"],
        default="gcn",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="regression",
        help="Type of prediction task.",
    )

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Optimizer weight decay.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension size.")
    parser.add_argument("--num-node-features", type=int, default=6, help="Number of input node features.")
    parser.add_argument("--num-edge-features", type=int, default=3, help="Number of input edge features (used by GIN).")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of output classes/targets.")

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data used for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data used for validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )

    # Miscellaneous
    parser.add_argument(
        "--save-path",
        type=str,
        default="molecule_gnn_final.pt",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def validate_splits(train_ratio: float, val_ratio: float) -> None:
    """Ensure train/validation ratios are sensible."""
    if not 0 < train_ratio < 1:
        raise ValueError("train-ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("val-ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio must be less than 1 to leave room for the test split.")


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    validate_splits(args.train_ratio, args.val_ratio)
    set_seed(args.seed)

    model_config = ModelConfig(
        num_node_features=args.num_node_features,
        num_edge_features=args.num_edge_features,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    )

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        task=args.task,
        num_workers=args.num_workers,
    )

    print("=" * 70)
    print("Molecular Property Prediction with GNNs")
    print("=" * 70)

    print("\n1. Loading dataset...")
    dataset = load_dataset(
        str(dataset_path),
        smiles_col=args.smiles_col,
        target_col=args.target_col,
    )
    print(f"   Loaded {len(dataset)} molecules from {dataset_path}")

    print("\n2. Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_size = sum(batch.num_graphs for batch in train_loader)
    val_size = sum(batch.num_graphs for batch in val_loader)
    test_size = sum(batch.num_graphs for batch in test_loader)
    print(f"   Train: {train_size} molecules")
    print(f"   Val:   {val_size} molecules")
    print(f"   Test:  {test_size} molecules")

    print("\n3. Initializing model...")
    if args.model == "gcn":
        model = SimpleMoleculeGCN(
            num_node_features=model_config.num_node_features,
            hidden_dim=model_config.hidden_dim,
            num_classes=model_config.num_classes,
            dropout=model_config.dropout,
            num_layers=model_config.num_layers,
        )
    else:
        model = AdvancedMoleculeGNN(
            num_node_features=model_config.num_node_features,
            num_edge_features=model_config.num_edge_features,
            hidden_dim=model_config.hidden_dim,
            num_classes=model_config.num_classes,
            dropout=model_config.dropout,
        )
    num_params = count_parameters(model)
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Parameters: {num_params:,}")
    print(f"   Device: {training_config.device}")
    
    # Show GPU optimization status
    import torch
    if torch.backends.mps.is_available():
        print("   GPU: Apple Silicon (MPS) - Mixed precision enabled")
    elif torch.cuda.is_available():
        print("   GPU: CUDA - Mixed precision enabled")
    else:
        print("   Device: CPU")

    print("\n4. Training model...")
    print("-" * 70)
    trainer = ModelTrainer(model, training_config)
    trainer.fit(train_loader, val_loader, task=args.task)

    print("\n5. Saving model...")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(save_path), model_config)
    print(f"   Model saved to {save_path}")

    print("\n6. Evaluating on test set...")
    print("-" * 70)
    evaluator = ModelEvaluator(model, training_config.device)
    if training_config.task == "classification":
        metrics: Dict[str, float] = evaluator.evaluate_classification(test_loader)
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    else:
        metrics = evaluator.evaluate_regression(test_loader)
        print(f"   MSE:  {metrics['mse']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE:  {metrics['mae']:.4f}")
        print(f"   R²:   {metrics['r2']:.4f}")

    print("\n7. Generating visualizations...")
    plot_training_history(trainer.history, save_path="training_history.png")
    if training_config.task == "classification":
        evaluator.plot_confusion_matrix(test_loader, save_path="confusion_matrix.png")
        evaluator.plot_roc_curve(test_loader, save_path="roc_curve.png")
        evaluator.plot_prediction_distribution(test_loader, save_path="pred_dist.png")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()