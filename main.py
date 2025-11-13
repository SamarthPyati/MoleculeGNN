#!.venv/bin/python3
from typing import Dict

from core.utils import (
    load_dataset, 
    set_seed, 
    create_data_loaders, 
    count_parameters, 
    save_model
)
from core.dataset import MoleculeDataset
from core.trainer import ModelTrainer
from core.evaluator import (
    ModelEvaluator, 
    plot_training_history
)
from models import SimpleMoleculeGCN
from config import ModelConfig, TrainingConfig

def main() -> None:
    """Main training pipeline with type hints"""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    model_config = ModelConfig(
        num_node_features=6,
        hidden_dim=128,
        num_classes=1,
        num_layers=3,
        dropout=0.3
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        weight_decay=1e-5,
        epochs=100,
        patience=15,
        task='classification'
    )
    
    print("=" * 70)
    print("Molecular Property Prediction with GNNs")
    print("=" * 70)
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    dataset: MoleculeDataset = load_dataset(
        'data/raw/tox21.csv',
        smiles_col='smiles',
        target_col='SR-HSE'
    )
    print(f"   Loaded {len(dataset)} molecules")
    
    # 2. Create data loaders
    print("\n2. Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers
    )
    # Calculate dataset sizes (DataLoader doesn't have .dataset attribute for lists)
    train_size = sum(batch.num_graphs for batch in train_loader)
    val_size = sum(batch.num_graphs for batch in val_loader)
    test_size = sum(batch.num_graphs for batch in test_loader)
    print(f"   Train: {train_size} molecules")
    print(f"   Val:   {val_size} molecules")
    print(f"   Test:  {test_size} molecules")
    
    # 3. Initialize model
    print("\n3. Initializing model...")
    model: SimpleMoleculeGCN = SimpleMoleculeGCN(
        num_node_features=model_config.num_node_features,
        hidden_dim=model_config.hidden_dim,
        num_classes=model_config.num_classes,
        dropout=model_config.dropout
    )
    
    num_params: int = count_parameters(model)
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Parameters: {num_params:,}")
    print(f"   Device: {training_config.device}")
    
    # 4. Train model
    print("\n4. Training model...")
    print("-" * 70)
    trainer = ModelTrainer(model, training_config)
    trainer.fit(train_loader, val_loader)
    
    # 5. Evaluate on test set
    print("\n5. Evaluating on test set...")
    print("-" * 70)
    
    evaluator = ModelEvaluator(model, training_config.device)
    
    if training_config.task == 'classification':
        classification_metrics: Dict[str, float] = evaluator.evaluate_classification(test_loader)
        print(f"   Accuracy:  {classification_metrics['accuracy']:.4f}")
        print(f"   Precision: {classification_metrics['precision']:.4f}")
        print(f"   Recall:    {classification_metrics['recall']:.4f}")
        print(f"   F1 Score:  {classification_metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {classification_metrics['roc_auc']:.4f}")
    else:
        regression_metrics: Dict[str, float] = evaluator.evaluate_regression(test_loader)
        print(f"   MSE:  {regression_metrics['mse']:.4f}")
        print(f"   RMSE: {regression_metrics['rmse']:.4f}")
        print(f"   MAE:  {regression_metrics['mae']:.4f}")
        print(f"   R²:   {regression_metrics['r2']:.4f}")
    
    # 6. Visualizations
    print("\n6. Generating visualizations...")
    plot_training_history(trainer.history, save_path='training_history.png')
    
    if training_config.task == 'classification':
        evaluator.plot_confusion_matrix(test_loader, save_path='confusion_matrix.png')
        evaluator.plot_roc_curve(test_loader, save_path='roc_curve.png')
        evaluator.plot_prediction_distribution(test_loader, save_path='pred_dist.png')
    
    # 7. Save model
    print("\n7. Saving model...")
    save_model(model, 'molecule_gcn_final.pt', model_config)
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()