import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import copy

# --- 1. Configuration ---

DATA_DIR = 'data/'  # Folder containing test_data.npy and train_data.npy
NUM_CLASSES = 10    # Digits 0-9
EPOCHS_PER_EVAL = 1 # Training epochs during GA
FINAL_EPOCHS = 30   # Training epochs for the best selected model
LEARNING_RATE = 0.01 
GPU_ID = 4

# GA Hyperparameters
POPULATION_SIZE = 10
GENERATIONS = 5     
MUTATION_RATE = 0.1
BATCH_RANGE = (16, 1024) 
ACTIVATIONS = ['relu', 'sigmoid', 'tanh'] 

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
        # Ensure float32 and normalize
        self.images = torch.tensor(np.array(images), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_data_flat():
    """
    Loads data using HW3 structure, flattens clients, filters for digits (0-9),
    and performs 80/20 train/val split.
    """
    print("Loading data...")
    train_raw = np.load(os.path.join(DATA_DIR, 'train_data.npy'), allow_pickle=True)
    test_raw = np.load(os.path.join(DATA_DIR, 'test_data.npy'), allow_pickle=True)

    # 1. Collect all data from "clients" into one big array
    all_train_images = []
    all_train_labels = []

    # Iterate through all clients in the loaded dictionary list
    for client_id in range(len(train_raw)):
        # FIX: Explicitly convert to numpy array to allow boolean indexing
        imgs = np.array(train_raw[client_id]['images'])
        lbls = np.array(train_raw[client_id]['labels'])
        
        # Filter for digits only (labels 0-9)
        mask = lbls < 10
        if np.sum(mask) > 0:
            all_train_images.append(imgs[mask])
            all_train_labels.append(lbls[mask])

    # Concatenate all arrays
    X_full = np.concatenate(all_train_images)
    y_full = np.concatenate(all_train_labels)

    # Normalize to 0-1 range
    X_full = X_full / 255.0

    # 2. Split 80% Train, 20% Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # 3. Process Test Data (Digits only)
    test_imgs_raw = np.array(test_raw[0]['images'])
    test_lbls_raw = np.array(test_raw[0]['labels'])
    test_mask = test_lbls_raw < 10
    X_test = (test_imgs_raw[test_mask]) / 255.0
    y_test = test_lbls_raw[test_mask]

    print(f"Data Loaded: Train {len(y_train)}, Val {len(y_val)}, Test {len(y_test)}")
    print(f"DEBUG: Unique labels in y_train: {np.unique(y_train)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- 4. Training Helper Functions ---

def train_and_evaluate_model(batch_size, activation_name, X_train, y_train, X_val, y_val, epochs):
    """
    Trains a model with specific hyperparameters and returns Validation F1 Score.
    """
    # Create DataLoaders for this specific batch size
    train_ds = EMNISTDataset(X_train, y_train)
    val_ds = EMNISTDataset(X_val, y_val)
    
    # SGD optimizer with mini-batches
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    model = Net(activation_name).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Training Loop
    for e in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Validation Evaluation (Macro F1)
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
    
    # Calculate Macro F1
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    return val_f1

# --- 5. Genetic Algorithm Implementation ---

def create_individual():
    """Create a random individual: [batch_size, activation_index]"""
    bs = random.randint(BATCH_RANGE[0], BATCH_RANGE[1])
    act_idx = random.randint(0, len(ACTIVATIONS) - 1)
    return [bs, act_idx]

def decode_individual(ind):
    """Convert indices to actual values"""
    return ind[0], ACTIVATIONS[ind[1]]

def calculate_fitness(population, X_train, y_train, X_val, y_val):
    fitness_scores = []
    print(f"  Evaluating population of {len(population)}...")
    for i, ind in enumerate(population):
        bs, act = decode_individual(ind)
        # Train briefly to check fitness
        f1 = train_and_evaluate_model(bs, act, X_train, y_train, X_val, y_val, EPOCHS_PER_EVAL)
        fitness_scores.append(f1)
    return fitness_scores

def roulette_selection(population, fitness_scores):
    """
    Selects a parent using Roulette Wheel Rule.
    """
    total_fit = sum(fitness_scores)
    if total_fit == 0:
        return random.choice(population)
        
    pick = random.uniform(0, total_fit)
    current = 0
    for i, individual in enumerate(population):
        current += fitness_scores[i]
        if current > pick:
            return individual
    return population[-1]

def crossover(p1, p2):
    """
    One-point crossover.
    """
    # Child 1: P1 Batch, P2 Activation
    c1 = [p1[0], p2[1]]
    # Child 2: P2 Batch, P1 Activation
    c2 = [p2[0], p1[1]]
    return c1, c2

def mutate(individual):
    """Randomly mutates genes based on rate"""
    # Mutate Batch Size
    if random.random() < MUTATION_RATE:
        individual[0] = random.randint(BATCH_RANGE[0], BATCH_RANGE[1])
    # Mutate Activation
    if random.random() < MUTATION_RATE:
        individual[1] = random.randint(0, len(ACTIVATIONS) - 1)
    return individual

# --- 6. Main Execution ---

def main():
    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_flat()

    # 2. GA Initialization
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    avg_fitness_history = []
    max_fitness_history = []
    best_overall_individual = None
    best_overall_fitness = -1

    print("\n--- Starting Genetic Algorithm ---")
    
    for gen in range(GENERATIONS):
        # Calc fitness
        fitness_scores = calculate_fitness(population, X_train, y_train, X_val, y_val)
        
        # Track stats
        gen_avg = np.mean(fitness_scores)
        gen_max = np.max(fitness_scores)
        avg_fitness_history.append(gen_avg)
        max_fitness_history.append(gen_max)
        
        # Track best found so far
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_overall_fitness:
            best_overall_fitness = fitness_scores[best_idx]
            best_overall_individual = copy.deepcopy(population[best_idx])

        print(f"Gen {gen+1}/{GENERATIONS} | Max F1: {gen_max:.4f} | Avg F1: {gen_avg:.4f}")
        print(f"  Best in Gen: BS={population[best_idx][0]}, Act={ACTIVATIONS[population[best_idx][1]]}")

        # Selection (Age Based - Full Replacement)
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = roulette_selection(population, fitness_scores)
            parent2 = roulette_selection(population, fitness_scores)
            
            child1, child2 = crossover(parent1, parent2)
            
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population[:POPULATION_SIZE]

    # --- 7. Final Results & Plotting ---
    
    # Plot GA Progress
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, GENERATIONS+1), avg_fitness_history, label='Average Fitness')
    plt.plot(range(1, GENERATIONS+1), max_fitness_history, label='Highest Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Validation F1 Score')
    plt.title('Genetic Algorithm Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('ga_fitness_progress.png')
    print("\nGA Plot saved to ga_fitness_progress.png")

    # Final Training
    best_bs, best_act_name = decode_individual(best_overall_individual)
    print(f"\n--- Best Hyperparameters Found ---")
    print(f"Batch Size: {best_bs}")
    print(f"Activation: {best_act_name}")

    print(f"\nTraining final model for {FINAL_EPOCHS} epochs...")
    
    train_ds = EMNISTDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=int(best_bs), shuffle=True)
    
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
        
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')
        training_f1_scores.append(epoch_f1)
        print(f"Epoch {e+1}: Train F1 = {epoch_f1:.4f}")

    # Plot Training F1
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, FINAL_EPOCHS+1), training_f1_scores, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training F1 Score')
    plt.title(f'Final Model Training (B={best_bs}, Act={best_act_name})')
    plt.grid(True)
    plt.savefig('final_training_f1.png')
    print("Training Plot saved to final_training_f1.png")

    # Evaluate on Test Data
    test_ds = EMNISTDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=int(best_bs), shuffle=False)
    
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