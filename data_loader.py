import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
def get_data_loaders(data_dir, batch_size=32):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found. Please download the Trashnet dataset and place it in '{data_dir}'.")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    class_names = dataset.classes
    return data_loader, class_names
if __name__ == "__main__":
    try:
        loader, classes = get_data_loaders("./dataset")
        print("Classes found:", classes)
    except FileNotFoundError as e:
        print(e)
