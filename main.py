#!.venv/bin/python3
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from utils import MoleculeDataset, ModelTrainer
from models import SimpleMoleculeGCN

def main():
    # Load dataset
    print("Loading dataset...")
    dataset = MoleculeDataset('data/raw/ESOL.csv', smiles_col='smiles', target_col='measured log solubility in mols per litre')
    
    # Remove None values (invalid molecules)
    dataset = [data for data in dataset if data is not None]
    
    # Split data (train, test, val)
    train_dataset, temp = train_test_split(dataset, test_size=0.3, random_state=42)
    # Split half of temp into test and validation
    val_dataset, test_dataset = train_test_split(temp, test_size=0.5, random_state=42)
    
    BATCH_SIZE: int = 32

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Init model
    num_node_features = 6  # Based on get_atom_features (grabbed only 6 feature, have to increase)
    model = SimpleMoleculeGCN(
        num_node_features=num_node_features,
        hidden_dim=128,
        num_classes=1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = ModelTrainer(model)
    trainer.fit(
        train_loader, 
        val_loader, 
        epochs=5, 
        lr=0.001,
        task='classification',
        patience=15
    )
    
    # Evaluate on test set
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, test_metric, metric_name = trainer.evaluate(test_loader, criterion, 'classification')
    print(f'\nTest Results: Loss: {test_loss:.4f}, {metric_name}: {test_metric:.4f}')
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.history['train_loss'], label='Train Loss')
    plt.plot(trainer.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    plt.subplot(1, 2, 2)
    plt.plot(trainer.history['val_metric'], label='Val ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.legend()
    plt.title('Validation Performance')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == '__main__':
    main()