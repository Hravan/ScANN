import numpy as np
from scann import ScANN

# Generate data
np.random.seed(2137)
data = np.random.randn(10000, 64)
query = np.random.randn(64)

# Initialize and train
ann = ScANN(num_clusters=100, top_clusters=5, top_k=10)
ann.fit(data)

# Search
neighbors = ann.search(query)
print("Approximate Nearest Neighbors:\n", neighbors)
