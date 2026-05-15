import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):
    """
    Creates and returns a PyTorch DataLoader for the Trashnet dataset.
    The images are resized to 224x224 and normalized for ResNet.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found. Please download the Trashnet dataset and place it in '{data_dir}'.")

    # ResNet-18 expects 224x224 images and ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset using ImageFolder (expects a directory of classes containing images)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # DataLoader without shuffling because we want to extract features and keep track of labels easily
    # We can also shuffle if we just want to train the unsupervised model, but keeping order helps
    # in creating a direct mapping array
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    class_names = dataset.classes

    return data_loader, class_names

if __name__ == "__main__":
    # Small test
    try:
        loader, classes = get_data_loaders("./dataset")
        print("Classes found:", classes)
    except FileNotFoundError as e:
        print(e)
