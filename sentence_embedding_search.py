# sentence_search_example.py
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

from product_quantizer import ProductQuantizer

# Step 1: Define your sentences
sentences = [
    "I love dogs.",
    "Cats are great pets.",
    "The sky is blue.",
    "She enjoys painting.",
    "Python is a programming language.",
    "Birds fly in the sky.",
    "He likes football.",
    "Programming in JavaScript is fun.",
    "The ocean is deep.",
    "Elephants are the largest land animals."
]

# Step 2: Load a small sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Encode the sentences
embeddings = model.encode(sentences, convert_to_numpy=True)

# Step 4: Reduce dimensionality for simplicity/visualization
pca = PCA(n_components=6)
reduced_embeddings = pca.fit_transform(embeddings)

# Step 5: Fit PQ
pq = ProductQuantizer(num_subvectors=3, num_centroids=4)
pq.fit(reduced_embeddings)
codes = pq.encode(reduced_embeddings)

# Step 6: Query
query_sentence = "I like animals."
query_embedding = model.encode([query_sentence], convert_to_numpy=True)
query_reduced = pca.transform(query_embedding)[0]
dists = pq.adc_distance(query_reduced, codes)

# Step 7: Rank and print results
sorted_indices = np.argsort(dists)
print(f"Query: {query_sentence}\n")
print("Nearest matches:")
for idx in sorted_indices[:5]:
    print(f"- {sentences[idx]} (distance: {dists[idx]:.3f})")

# Step 8: Generate subvector group plots (not shown)
for i in range(pq.n_subvectors):
    start = i * pq.subvector_dim
    end = (i + 1) * pq.subvector_dim
    subspace_data = reduced_embeddings[:, start:end]
    query_sub = query_reduced[start:end]
    centroids = pq.codebooks[i]

    # Assign each point to its cluster for coloring
    cluster_assignments = codes[:, i]
    cluster_colors = [f"C{cluster}" for cluster in cluster_assignments]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(subspace_data[:, 0], subspace_data[:, 1], c=cluster_colors, s=80, label='Sentences')
    for idx, (x, y) in enumerate(subspace_data):
        ax.annotate(f"{idx}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color='black')

    # Color centroids based on majority color of assigned points
    centroid_colors = []
    for j in range(len(centroids)):
        members = [cluster_colors[k] for k in range(len(cluster_assignments)) if cluster_assignments[k] == j]
        color = members[0] if members else 'gray'
        centroid_colors.append(color)

    ax.scatter(centroids[:, 0], centroids[:, 1], c=centroid_colors, marker='X', s=100, label='Centroids')
    ax.scatter(query_sub[0], query_sub[1], c='green', marker='*', s=150, label='Query')

    for j, centroid in enumerate(centroids):
        ax.plot([query_sub[0], centroid[0]], [query_sub[1], centroid[1]], 'k--', alpha=0.3)
        mid_x, mid_y = (query_sub[0] + centroid[0]) / 2, (query_sub[1] + centroid[1]) / 2
        dist = np.linalg.norm(query_sub - centroid)
        ax.text(mid_x, mid_y, f"{dist:.2f}", fontsize=8, color='black')

    legend_text = "\n".join([f"{i}: {s}" for i, s in enumerate(sentences)])
    fig.text(1.02, 0.5, legend_text, fontsize=8, va='center', transform=ax.transAxes)

    # Static legend icons with fixed colors
    custom_legend = [
        Line2D([0], [0], marker='o', color='gray', label='Sentences', linestyle='', markersize=8),
        Line2D([0], [0], marker='X', color='gray', label='Centroids', linestyle='', markersize=8),
        Line2D([0], [0], marker='*', color='green', label='Query', linestyle='', markersize=12)
    ]
    ax.legend(handles=custom_legend)

    ax.set_title(f"Subvector Group {i + 1} (Cluster Color-coded)")
    ax.grid(True)
    fig.savefig(f"subvector_group_{i + 1}.png")
    plt.close(fig)
