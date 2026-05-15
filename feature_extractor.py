import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        # Load pre-trained ResNet-18
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final fully connected layer to use the network as a feature extractor
        # The output of the layer before fc is 512 dimensions for ResNet-18
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)

def extract_features(data_loader, device='cuda'):
    """
    Passes all images in the data_loader through the feature extractor.
    Returns the features and true labels.
    """
    model = ResNetFeatureExtractor().to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    
    print("Extracting features using ResNet-18...")
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = model(images)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
            
    # Concatenate all batches into a single tensor
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels
