import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.loader import MyLoader
from model.office_model import OfficeModel
from planb.specific_loader import domain_loader


class PseudoLabelDataset(Dataset):
    def __init__(self, pseudo_labels):
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.pseudo_labels)

    def __getitem__(self, idx):
        image, label, confidence = self.pseudo_labels[idx]
        return image, label


def evaluate_model_per_class(model, data_loader, num_classes=65):
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    class_accuracy = class_correct / class_total
    return class_accuracy


def generate_pseudo_labels(model, data_loader, threshold=0.5):
    pseudo_labels = []
    class_counts = torch.zeros(65, dtype=torch.int)

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _, outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            max_probs, predicted_labels = torch.max(probabilities, dim=1)
            for i in range(len(max_probs)):
                if max_probs[i] >= threshold:
                    pseudo_labels.append((inputs[i].cpu(), predicted_labels[i].cpu(), max_probs[i].cpu()))
                    class_counts[predicted_labels[i]] += 1

    return pseudo_labels, class_counts


src, tar = 'Art', 'Product'
src_train, src_test = domain_loader(src, batch_size=16, use_transforms=True)
tar_train, tar_test = domain_loader(tar, batch_size=16, use_transforms=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OfficeModel(65).to(device)
model.load_state_dict(torch.load('best_model_on_src.pth'))
model.eval()

unfilled = tar_train

class_accuracy = evaluate_model_per_class(model, unfilled)
for class_idx, accuracy in enumerate(class_accuracy):
    print(f'Class {class_idx}: Accuracy {accuracy:.4f}')

pseudo_labels, class_counts = generate_pseudo_labels(model, unfilled)
for class_idx, count in enumerate(class_counts):
    print(f'Class {class_idx}: {count} images')

pseudo_label_dataset = PseudoLabelDataset(pseudo_labels)
pseudo_label_loader = MyLoader(pseudo_label_dataset, domain=tar, batch_size=16, shuffle=True)
print(f'Generated {len(pseudo_labels)} pseudo-labels.')
