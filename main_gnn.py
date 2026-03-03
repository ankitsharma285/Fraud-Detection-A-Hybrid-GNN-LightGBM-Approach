import yaml
import pandas as pd
import numpy as np

import torch 
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import NeighborLoader

from pathlib import Path 

import helper
import models
import engine

import gc 

def main():
    config_path = "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset_params']
    file_train_transaction = Path(dataset_config['data_path']) / 'train_transaction.csv'
    file_train_identity = Path(dataset_config['data_path']) / 'train_identity.csv'
    
    train_trans = pd.read_csv(file_train_transaction)
    train_id = pd.read_csv(file_train_identity)

    # Merge on TransactionID
    train = pd.merge(train_trans, train_id, on='TransactionID', how='left')
    del train_trans, train_id

    # Basic cleaning: Fill NAs in categorical columns with 'missing'
    cat_cols = train.select_dtypes(include=['object']).columns
    train[cat_cols] = train[cat_cols].fillna('missing')

    # Create a unique 'Card_ID' by combining card features
    train['card_id'] = train['card1'].astype(str) + "_" + train['card2'].astype(str)

    # Generate mappings for our entity nodes
    card_map, num_cards = helper.create_mapping(train['card_id'])
    dev_map, num_devices = helper.create_mapping(train['DeviceInfo'])

    # Map the strings in the dataframe to these new integer IDs
    train['card_node_idx'] = train['card_id'].map(card_map)
    train['dev_node_idx'] = train['DeviceInfo'].map(dev_map)

    print("Creating group aggregates...")
    print("Calculating group statistics (Memory Lean)...")
    card_stats = train.groupby('card_id')['TransactionAmt'].agg(['mean', 'std']).fillna(0)
    card_counts = train['card_id'].value_counts()

    mean_map = card_stats['mean'].to_dict()
    std_map = card_stats['std'].to_dict()
    count_map = card_counts.to_dict()

    del card_stats, card_counts
    gc.collect()

    # Map the values back to the main DataFrame
    print("Mapping stats back to main DataFrame...")
    train['card_amt_mean'] = train['card_id'].map(mean_map).astype('float32')
    train['card_amt_std'] = train['card_id'].map(std_map).astype('float32')
    train['card_counts'] = train['card_id'].map(count_map).astype('int32')

    # Calculate Time Delta manually 
    train = train.sort_values(['card_id', 'TransactionDT'])
    train['TransactionDT_diff'] = train.groupby('card_id')['TransactionDT'].diff().fillna(0).astype('float32')
    train = train.sort_index()

    del mean_map, std_map, count_map
    gc.collect()
    
    # Define the new aggregate feature list
    agg_cols = ['card_amt_mean', 'card_amt_std', 'card_counts', 'TransactionDT_diff']

    # FEATURE SCALING 
    # Log transform amount
    train['Amt_Scaled'] = np.log1p(train['TransactionAmt'])

    # Scale aggregates for the GNN
    v_cols = [c for c in train.columns if c.startswith('V')]
    feature_to_scale = v_cols + agg_cols

    scaler = StandardScaler()
    train[feature_to_scale] = scaler.fit_transform(train[feature_to_scale].fillna(0))

    # --- BUILD HETERODATA ---
    # Update feature_list 
    feature_list = ['Amt_Scaled'] + v_cols + agg_cols
    
    data = HeteroData()
    data['transaction'].x = torch.tensor(train[feature_list].values, dtype=torch.float)
    # data['transaction].x shape [no_transactions, 340 Features]
    data['transaction'].y = torch.tensor(train['isFraud'].values, dtype=torch.long)

    # Add edges 
    # Add TRANSACTION -> CARD edges
    data['transaction', 'to', 'card'].edge_index = torch.stack([
        torch.arange(len(train)), 
        torch.tensor(train['card_node_idx'].values)
    ], dim=0)

    # Add TRANSACTION -> DEVICE edges 
    data['transaction', 'to', 'device'].edge_index = torch.stack([
        torch.arange(len(train)), 
        torch.tensor(train['dev_node_idx'].values)
    ], dim=0)

    # Make the graph Undirected (Creates rev_to edges)
    data = T.ToUndirected()(data)

    # Initialize Entity Features
    # Use the num_cards/num_devices variables from the mapping step
    data['card'].x = torch.ones((num_cards, 16))
    data['device'].x = torch.ones((num_devices, 16))

    num_transactions = data['transaction'].x.shape[0]
    indices = torch.randperm(num_transactions)
    train_idx = indices[:int(num_transactions * 0.8)]
    val_idx = indices[int(num_transactions * 0.8):]

    # Loader for Training 
    train_loader = NeighborLoader(
        data,
        num_neighbors=[25, 15], 
        batch_size=1024,
        input_nodes=('transaction', train_idx),
        shuffle=True
    )

    # Loader for Validation 
    val_loader = NeighborLoader(
        data,
        num_neighbors=[20, 20],
        batch_size=2048,
        input_nodes=('transaction', val_idx),
        shuffle=False
    )

    
    # Define the model as if it's a simple graph, 
    # then 'to_hetero' automatically replicates it for our specific node/edge types.
    model = models.GNN(hidden_channels=128, out_channels=2)
    model = to_hetero(model, data.metadata(), aggr='sum')

    # Calculate the ratio of legit to fraud
    num_legit = (data['transaction'].y == 0).sum().item()
    num_fraud = (data['transaction'].y == 1).sum().item()

    # Weight for the 'Fraud' class (Class 1)
    # Typically: total_samples / (num_classes * num_class_samples)
    if num_fraud > 0:
        fraud_weight = num_legit / num_fraud
    else:
        fraud_weight = 1.0
    
    # Create the weight tensor: [Weight for Legit, Weight for Fraud]
    weights = torch.tensor([1.0, fraud_weight], dtype=torch.float)
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    weights = weights.to(device)
    criterion = criterion.to(device)
    best_pr_auc = 0.0
    checkpoint_path = "best_gnn_model_128.pt"


    for epoch in range(1, 51): 
        loss = engine.train_full(model, optimizer,
                                 criterion, train_loader, 
                                 device)
        val_pr_auc = engine.evaluate_full(model, device, val_loader)
        scheduler.step(val_pr_auc)
    
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val PR-AUC: {val_pr_auc:.4f}')
    
        # Checkpoint: Save if this is the best model so far
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pr_auc': best_pr_auc,
            }, checkpoint_path)
            print(f"--- New Best Model Saved (PR-AUC: {best_pr_auc:.4f}) ---")


if __name__ == '__main__':
    main()