import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

def evaluate_clustering(true_labels, predicted_clusters, class_names, output_dir="./results"):
    """
    Evaluates the clustering by plotting a bar chart of actual classes vs. grouped clusters.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    true_counts = Counter(true_labels.numpy())
    pred_counts = Counter(predicted_clusters.numpy())
    
    num_classes = len(class_names)
    
    # Sort counts by index to make sure they align
    true_distribution = [true_counts.get(i, 0) for i in range(num_classes)]
    pred_distribution = [pred_counts.get(i, 0) for i in range(num_classes)]
    
    # 1. Overall Distribution Comparison
    x = np.arange(num_classes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, true_distribution, width, label='True Elements per Class')
    
    # We label predicted clusters as "Cluster 0", "Cluster 1", etc. because they don't map directly
    # to the class names without a mapping function.
    rects2 = ax.bar(x + width/2, pred_distribution, width, label='Elements in Competitive Clusters')

    ax.set_ylabel('Number of Images')
    ax.set_title('Comparison: True Classes vs. Competitive Network Clusters')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'))
    print("Saved distribution comparison chart to results/distribution_comparison.png")
    plt.close()

    # 2. Composition of each cluster (Confusion Matrix equivalent)
    # We want to see what true classes ended up in each cluster
    cluster_compositions = {i: {j: 0 for j in range(num_classes)} for i in range(num_classes)}
    for true_l, pred_c in zip(true_labels.numpy(), predicted_clusters.numpy()):
        cluster_compositions[pred_c][true_l] += 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(num_classes):
        composition = [cluster_compositions[i][j] for j in range(num_classes)]
        axes[i].bar(class_names, composition, color='skyblue')
        axes[i].set_title(f'Cluster {i} Composition')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_compositions.png'))
    print("Saved cluster composition charts to results/cluster_compositions.png")
    plt.close()
