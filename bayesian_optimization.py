import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Configuration ---

DATA_DIR = 'data/'  # Folder containing test_data.npy and train_data.npy
NUM_CLASSES = 10    # Digits 0-9
EPOCHS_PER_EVAL = 30 # Optimized for RTX 2080 Super (higher accuracy for BayesOpt)
FINAL_EPOCHS = 10   # Training epochs for the best selected model
LEARNING_RATE = 0.01
GPU_ID = 3

# Optimization Ranges
# Activations mapped to indices: 0:relu, 1:sigmoid, 2:tanh
ACTIVATIONS = ['relu', 'sigmoid', 'tanh'] 
PBOUNDS = {
    'batch_size': (16, 1024),
    'activation_idx': (0, 2.999) # Upper bound just below 3 to map 0, 1, 2 safely
}
INIT_POINTS = 5   # Random exploration steps
N_ITER = 15       # Optimization steps (Increased slightly for better search)

# Check Device
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. Model Definition ---
class Net(torch.nn.Module):
    def __init__(self, activation_name):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        # Add Batch Normalization here
        self.bn1 = torch.nn.BatchNorm1d(128) 
        self.fc2 = torch.nn.Linear(128, NUM_CLASSES)
        
        if activation_name == 'relu':
            self.act = torch.nn.ReLU()
        elif activation_name == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif activation_name == 'tanh':
            self.act = torch.nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation_name}")

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.bn1(x) # Apply Batch Norm before Activation
        x = self.act(x)
        x = self.fc2(x)
        return x
    
# --- 3. Data Handling ---

class EMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(np.array(images), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_data_flat():
    """
    Loads data, flattens clients, filters for digits (0-9),
    and performs 80/20 train/val split.
    """
    print("Loading data...")
    train_raw = np.load(os.path.join(DATA_DIR, 'train_data.npy'), allow_pickle=True)
    test_raw = np.load(os.path.join(DATA_DIR, 'test_data.npy'), allow_pickle=True)

    # 1. Collect all data from "clients" into one big array
    all_train_images = []
    all_train_labels = []

    for client_id in range(len(train_raw)):
        # FIX: Explicitly convert to numpy array to allow boolean indexing
        imgs = np.array(train_raw[client_id]['images'])
        lbls = np.array(train_raw[client_id]['labels'])
        
        # Filter for digits only (labels 0-9)
        mask = lbls < 10
        if np.sum(mask) > 0:
            all_train_images.append(imgs[mask])
            all_train_labels.append(lbls[mask])

    # Concatenate and Normalize
    X_full = np.concatenate(all_train_images) / 255.0
    y_full = np.concatenate(all_train_labels)

    # 2. Split 80% Train, 20% Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # 3. Process Test Data (Digits only)
    # FIX: Explicitly convert to numpy array here as well
    test_imgs_raw = np.array(test_raw[0]['images'])
    test_lbls_raw = np.array(test_raw[0]['labels'])
    
    test_mask = test_lbls_raw < 10
    X_test = (test_imgs_raw[test_mask]) / 255.0
    y_test = test_lbls_raw[test_mask]

    print(f"Data Loaded: Train {len(y_train)}, Val {len(y_val)}, Test {len(y_test)}")
    print(f"DEBUG: Unique labels in y_train: {np.unique(y_train)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- 4. Black Box Function ---

# Global variables for data (needed because bayes_opt function signature is fixed)
X_TRAIN, Y_TRAIN = None, None
X_VAL, Y_VAL = None, None

def evaluate_network(batch_size, activation_idx):
    """
    The black-box function to maximize.
    Inputs are floats from the optimizer.
    """
    # 1. Cast parameters to integers/types
    bs = int(batch_size)
    act_i = int(activation_idx)
    
    # SAFETY CLAMP: Ensure index is within [0, 2] even if optimizer hits boundary
    act_i = max(0, min(act_i, len(ACTIVATIONS) - 1))
    activation_name = ACTIVATIONS[act_i]

    # 2. Setup DataLoaders
    train_ds = EMNISTDataset(X_TRAIN, Y_TRAIN)
    val_ds = EMNISTDataset(X_VAL, Y_VAL)
    
    # --- FIX: drop_last=True prevents crash on batch size 1 ---
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # 3. Initialize Model and Optimizer
    model = Net(activation_name).to(DEVICE)
    # Ensure you are using the momentum fix from before
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # 4. Train (Mini-evaluation)
    for e in range(EPOCHS_PER_EVAL):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # 5. Validate and return F1
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    return val_f1

# --- 5. Main Execution ---

def main():
    global X_TRAIN, Y_TRAIN, X_VAL, Y_VAL
    
    # 1. Load Data
    X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, X_test, y_test = load_data_flat()

    print("\n--- Starting Bayesian Optimization ---")
    
    # 2. Initialize Optimizer
    optimizer = BayesianOptimization(
        f=evaluate_network,
        pbounds=PBOUNDS,
        random_state=42,
        verbose=2
    )

    # 3. Run Optimization
    # init_points: Steps of random exploration before Gaussian Process starts
    # n_iter: Steps of Bayesian Optimization
    optimizer.maximize(
        init_points=INIT_POINTS,
        n_iter=N_ITER,
    )

    # 4. Extract Best Results
    best_params = optimizer.max['params']
    best_f1 = optimizer.max['target']
    
    best_bs = int(best_params['batch_size'])
    best_act_idx = int(best_params['activation_idx'])
    # Safety clamp again for final extraction
    best_act_idx = max(0, min(best_act_idx, len(ACTIVATIONS) - 1))
    best_act_name = ACTIVATIONS[best_act_idx]

    print(f"\n--- Best Hyperparameters Found ---")
    print(f"Batch Size: {best_bs}")
    print(f"Activation: {best_act_name}")
    print(f"Validation F1 (approx): {best_f1:.4f}")

# 5. Train Final Model
    print(f"\nTraining final model for {FINAL_EPOCHS} epochs...")
    
    train_ds = EMNISTDataset(X_TRAIN, Y_TRAIN)
    
    # --- FIX: drop_last=True here as well ---
    train_loader = DataLoader(train_ds, batch_size=best_bs, shuffle=True, drop_last=True)
    
    final_model = Net(best_act_name).to(DEVICE)
    optimizer = torch.optim.SGD(final_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    training_f1_scores = []
    
    for e in range(FINAL_EPOCHS):
        final_model.train()
        all_preds, all_targets = [], []
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = final_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
        
        # Calculate training F1 for this epoch
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')
        training_f1_scores.append(epoch_f1)
        print(f"Epoch {e+1}: Train F1 = {epoch_f1:.4f}")

    # 6. Plot Training F1 vs Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, FINAL_EPOCHS+1), training_f1_scores, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training F1 Score')
    plt.title(f'Final Model Training (BayesOpt: B={best_bs}, Act={best_act_name})')
    plt.grid(True)
    plt.savefig('bayes_opt_training_f1.png')
    print("Training Plot saved to bayes_opt_training_f1.png")

    # 7. Evaluate on Test Data
    test_ds = EMNISTDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=best_bs, shuffle=False)
    
    final_model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = final_model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_f1 = f1_score(test_targets, test_preds, average='macro')
    print(f"\nFinal Test F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()