import torch 
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import torch.nn.functional as F 


def train(model, optimizer,
          data, criterion):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x_dict, data.edge_index_dict)
    
    # We only care about predicting 'transaction' nodes
    pred = out['transaction']
    target = data['transaction'].y
    
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_full(model, optimizer,
          criterion, train_loader, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x_dict, batch.edge_index_dict)
        
        batch_size = batch['transaction'].batch_size
        pred = out['transaction'][:batch_size]
        target = batch['transaction'].y[:batch_size]
        
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def train_50k(model, optimizer,
          data, criterion):
    model.train()
    optimizer.zero_grad()
    
    # Create a mask: 80% for training, 20% for validation
    num_transactions = data['transaction'].x.shape[0]
    indices = torch.randperm(num_transactions)

    train_idx = indices[:int(num_transactions * 0.8)]
    val_idx = indices[int(num_transactions * 0.8):]
    
    
    out = model(data.x_dict, data.edge_index_dict)
    
    # ONLY calculate loss on the training nodes
    loss = criterion(out['transaction'][train_idx], data['transaction'].y[train_idx])

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    
    # Get probabilities for the 'Fraud' class (index 1)
    probs = F.softmax(out['transaction'], dim=1)[:, 1].numpy()
    y_true = data['transaction'].y.numpy()
    
    precision, recall, _ = precision_recall_curve(y_true, probs)
    auc_score = auc(recall, precision)
    
    print(f'PR-AUC Score: {auc_score:.4f}')
    return precision, recall


@torch.no_grad()
def evaluate_full(model, device, val_loader):
    model.eval()
    all_probs, all_labels = [], []
    
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        
        batch_size = batch['transaction'].batch_size
        logits = out['transaction'][:batch_size]
        
        probs = F.softmax(logits, dim=1)[:, 1]
        labels = batch['transaction'].y[:batch_size]
        
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_scores = torch.cat(all_probs).numpy()
    return average_precision_score(y_true, y_scores)