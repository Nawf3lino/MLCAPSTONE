import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

# --- Flowchart Descriptions ---
# KNN: The user interaction matrix is used to compute a cosine similarity matrix. The data is split into train/test sets, and a KNN model (e.g., from scikit-surprise) is trained and used to make predictions.
# NMF: The user interaction matrix is factorized using Non-negative Matrix Factorization (NMF) into two dense matrices (user and item latent factors). Their product gives predicted scores for user-item pairs.
# Neural Network Embedding: User and course IDs are one-hot encoded, passed through embedding layers, concatenated, and fed into a neural network to predict scores.

# --- Data Preparation ---
ratings = pd.read_csv('ratings.csv')
user_encoder = LabelEncoder()
course_encoder = LabelEncoder()
ratings['user_enc'] = user_encoder.fit_transform(ratings['user'])
ratings['item_enc'] = course_encoder.fit_transform(ratings['item'])
num_users = ratings['user_enc'].nunique()
num_items = ratings['item_enc'].nunique()

# Train/test split
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# --- KNN Model ---
# Build user-item matrix for train
train_matrix = np.zeros((num_users, num_items))
for row in train.itertuples():
    train_matrix[row.user_enc, row.item_enc] = row.rating

# Fit KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(train_matrix)

# Predict for test set
knn_preds = []
for row in test.itertuples():
    user_vec = train_matrix[row.user_enc].reshape(1, -1)
    dists, idxs = knn.kneighbors(user_vec, n_neighbors=6)  # include self
    # Exclude self
    sim_users = idxs[0][1:]
    sim_ratings = train_matrix[sim_users, row.item_enc]
    pred = sim_ratings[sim_ratings > 0].mean() if np.any(sim_ratings > 0) else train_matrix[:, row.item_enc][train_matrix[:, row.item_enc] > 0].mean() if np.any(train_matrix[:, row.item_enc] > 0) else train['rating'].mean()
    knn_preds.append(pred)
knn_rmse = sqrt(mean_squared_error(test['rating'], knn_preds))

# --- NMF Model ---
R = train_matrix.copy()
model = NMF(n_components=20, init='random', random_state=42, max_iter=200)
U = model.fit_transform(R)
I = model.components_
R_pred = np.dot(U, I)
nmf_preds = [R_pred[row.user_enc, row.item_enc] for row in test.itertuples()]
nmf_rmse = sqrt(mean_squared_error(test['rating'], nmf_preds))

# --- Neural Network Embedding Model ---
class RecSysNN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        x = torch.cat([u, i], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

# Device selection for Apple Silicon (MPS), CUDA, or CPU
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()
model = RecSysNN(num_users, num_items).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Prepare data
train_users = torch.tensor(train['user_enc'].values, dtype=torch.long, device=device)
train_items = torch.tensor(train['item_enc'].values, dtype=torch.long, device=device)
train_ratings = torch.tensor(train['rating'].values, dtype=torch.float, device=device)

# Train NN
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    preds = model(train_users, train_items)
    loss = loss_fn(preds, train_ratings)
    loss.backward()
    optimizer.step()

# Predict for test set
model.eval()
test_users = torch.tensor(test['user_enc'].values, dtype=torch.long, device=device)
test_items = torch.tensor(test['item_enc'].values, dtype=torch.long, device=device)
with torch.no_grad():
    nn_preds = model(test_users, test_items).cpu().numpy()
nn_rmse = sqrt(mean_squared_error(test['rating'], nn_preds))

# --- Bar Chart of RMSEs ---
models = ['KNN', 'NMF', 'NeuralNet']
rmse_scores = [knn_rmse, nmf_rmse, nn_rmse]
plt.figure(figsize=(8, 5))
plt.bar(models, rmse_scores, color=['#A3C9A8', '#7BA69A', '#4B8673'])
plt.ylabel('RMSE')
plt.title('Performance Comparison of Collaborative Filtering Models')
for i, v in enumerate(rmse_scores):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('collabfilter_rmse_comparison.png', dpi=300)
plt.close()

# --- Brief Explanation of the Bar Chart ---
# The bar chart visualizes the RMSE (Root Mean Squared Error) of each collaborative filtering model. Lower RMSE indicates better prediction accuracy. This comparison helps identify which model performs best on the dataset. 