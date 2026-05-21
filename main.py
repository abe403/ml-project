import torch
import os
from data_loader import get_data_loaders
from feature_extractor import extract_features
from competitive_net import CompetitiveNetwork
from evaluation import evaluate_clustering
def main():
    DATASET_DIR = "./dataset"
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.05
    NUM_CLUSTERS = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print(f"Error: No dataset found in '{DATASET_DIR}'.")
        print("Please download the Trashnet dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/feyzazkefe/trashnet/data")
        print("And extract it so that 'cardboard', 'glass', etc. folders are inside './dataset/'.")
        return
    print("Initializing DataLoader...")
    data_loader, class_names = get_data_loaders(DATASET_DIR, batch_size=BATCH_SIZE)
    print(f"Found {len(class_names)} classes: {class_names}")
    features, true_labels = extract_features(data_loader, device=device)
    print(f"Extracted features shape: {features.shape}")
    net = CompetitiveNetwork(input_dim=512, num_clusters=NUM_CLUSTERS, learning_rate=LEARNING_RATE)
    net.train_network(features, epochs=EPOCHS)
    print("Clustering data...")
    features_norm = torch.norm(features, p=2, dim=1, keepdim=True)
    features_normalized = features / (features_norm + 1e-8)
    with torch.no_grad():
        predicted_clusters = net(features_normalized)
    print("Evaluating and plotting results...")
    evaluate_clustering(true_labels, predicted_clusters, class_names, output_dir="./results")
    print("Process completed successfully. Check the ./results/ folder.")
if __name__ == "__main__":
    main()
