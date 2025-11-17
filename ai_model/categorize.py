#!/usr/bin/env python3
import argparse
import os
#!/usr/bin/env python3
import argparse
import os
import sys
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F


def find_dataset_root():
    candidate = os.path.join(os.path.dirname(__file__), "dataset")
    if os.path.isdir(candidate):
        return candidate
    return "dataset"


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        good_imgs = []
        for path, label in self.samples:
            try:
                img = Image.open(path)
                img.verify()
                good_imgs.append((path, label))
            except Exception as e:
                print(f"❌ Removing bad file: {path} ({e})")
        self.samples = good_imgs
        self.imgs = self.samples

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"⚠️ Skipping bad file at index {index}: {self.imgs[index][0]} ({e})")
            return torch.zeros(3, 224, 224), 0


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def prepare_loaders(dataset_root, train_transform, test_transform, batch_size=16):
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dataset = SafeImageFolder(dataset_root, transform=test_transform)

    if len(dataset) < 2:
        raise RuntimeError("Not enough images in dataset to split into train/test")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    class_counts = [len([s for s in dataset.samples if s[1] == i]) for i in range(len(dataset.classes))]
    print("Class counts:", dict(zip(dataset.classes, class_counts)))

    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in train_dataset]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, train_loader, test_loader


def build_model(num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model.to(device)


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    batches = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    return 100 * correct / total if total > 0 else 0.0


def predict_image(model, dataset, img_path, test_transform, device, topk=3):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    img = test_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(img)
        probs = F.softmax(pred, dim=1)

        num_classes = probs.size(1)
        if num_classes == 0:
            print("No classes available to predict.")
            return

        k = min(topk, num_classes)
        top_p, top_class = torch.topk(probs, k, dim=1)

        print("\nTop Predictions:")
        for i in range(k):
            cls = dataset.classes[top_class[0][i].item()]
            conf = top_p[0][i].item() * 100
            print(f"  {cls}: {conf:.2f}%")

        print("\nAll probabilities:")
        for cls, p in zip(dataset.classes, probs[0]):
            print(f"  {cls}: {p.item()*100:.2f}%")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=find_dataset_root(), help="Path to dataset root")
    parser.add_argument("--image", default=None, help="Path to a single image to predict (skips training)")
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke test (no training) to verify model/load/predict")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate an existing model.pth on the dataset and print accuracy")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (if not smoke)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training/eval")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit number of training batches per epoch (for quick tests)")
    parser.add_argument("--max-test-batches", type=int, default=None, help="Limit number of test batches during evaluation (for quick tests)")
    args = parser.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_transform, test_transform = build_transforms()

    dataset, train_loader, test_loader = prepare_loaders(args.dataset, train_transform, test_transform, batch_size=args.batch)

    # If eval-only, load existing model and print accuracy, then exit
    if args.eval_only:
        ckpt_path = os.path.join(os.path.dirname(__file__), "model.pth")
        if not os.path.exists(ckpt_path):
            print(f"model.pth not found at {ckpt_path}. Train a model first.")
            return 2
        checkpoint = torch.load(ckpt_path, map_location=device)
        classes = checkpoint.get("classes", dataset.classes)
        model = build_model(num_classes=len(classes), device=device)
        model.load_state_dict(checkpoint["model_state"])  # type: ignore[index]
        test_acc = evaluate(model, test_loader, device, max_batches=args.max_test_batches)
        train_acc = evaluate(model, train_loader, device, max_batches=args.max_test_batches)
        print(f"Accuracy (test): {test_acc:.2f}%")
        print(f"Accuracy (train): {train_acc:.2f}%")
        return 0

    model = build_model(num_classes=len(dataset.classes), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Smoke test: run a single forward pass on a batch and optionally predict a provided image
    if args.smoke:
        print('\n=== SMOKE TEST ===')
        # get one batch
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                print('Smoke forward pass output shape:', outputs.shape)
            break

        if args.image:
            predict_image(model, dataset, args.image, test_transform, device)

        return 0

    if args.image:
        predict_image(model, dataset, args.image, test_transform, device)
        return 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        processed_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batches += 1
            if args.max_train_batches is not None and processed_batches >= args.max_train_batches:
                break

        scheduler.step()

        denom = processed_batches if processed_batches > 0 else len(train_loader)

        train_acc = evaluate(model, train_loader, device, max_batches=args.max_test_batches)
        test_acc = evaluate(model, test_loader, device, max_batches=args.max_test_batches)
        print(f"Epoch {epoch+1}, Loss: {running_loss/denom:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    save_path = os.path.join(os.path.dirname(__file__), "model.pth")
    torch.save({
        "model_state": model.state_dict(),
        "classes": dataset.classes
    }, save_path)

    print(f"\n✅ Model trained and saved as {save_path}")


if __name__ == '__main__':
    sys.exit(main())
