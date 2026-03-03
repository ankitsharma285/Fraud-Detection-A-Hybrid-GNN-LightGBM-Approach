import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import yaml
from pathlib import Path 

config_path = "config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

dataset_config = config['dataset_params']
file_train_transaction = Path(dataset_config['data_path']) / 'train_transaction.csv'
file_train_identity = Path(dataset_config['data_path']) / 'train_identity.csv'
    
train_trans = pd.read_csv(file_train_transaction)
train_id = pd.read_csv(file_train_identity)
train = pd.merge(train_trans, train_id, on='TransactionID', how='left')

train['card_id'] = train['card1'].astype(str) + "_" + train['card2'].astype(str)
train['Amt_Log'] = np.log1p(train['TransactionAmt'])

print("Creating group aggregates...")
train['card_id_amt_mean'] = train.groupby('card_id')['TransactionAmt'].transform('mean')
train['card_id_amt_std'] = train.groupby('card_id')['TransactionAmt'].transform('std')
train['card_id_count'] = train.groupby('card_id')['TransactionID'].transform('count')

# Handle Categoricals
# LightGBM handles strings better if converted to 'category' type
cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'card_id']
for col in cat_cols:
    train[col] = train[col].astype('category')

# Define Feature List
v_cols = [c for c in train.columns if c.startswith('V')]
features = ['Amt_Log', 'card_id_amt_mean', 'card_id_amt_std', 'card_id_count'] + v_cols + cat_cols

X = train[features]
y = train['isFraud']

# Train/Val Split (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create LightGBM Dataset (Memory efficient)
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

# Parameters (Optimized for PR-AUC and Imbalance)
params = {
    'objective': 'binary',
    'metric': 'average_precision', 
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'is_unbalance': True,          
    'random_state': 42,
    'verbosity': -1
}

# Train
print("Starting LightGBM Training...")
model = lgb.train(
    params,
    dtrain,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'valid'],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
)

# Final Evaluation
preds = model.predict(X_val)
pr_auc = average_precision_score(y_val, preds)
print(f"\nFinal LightGBM Val PR-AUC: {pr_auc:.4f}")