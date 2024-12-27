import torch
from tqdm import tqdm


def test_acc(model, dataloader, cls_loss, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    pbar = tqdm(dataloader, desc='Testing', ascii=True)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feature, outputs = model(images)
            loss = cls_loss(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            batch_size = labels.size(0)
            total_samples += batch_size
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item() * batch_size

            acc = 100 * total_correct / total_samples
            avg_loss = total_loss / total_samples
            pbar.set_postfix({
                'Acc': f'{acc:.2f}%',
                'Loss': f'{avg_loss:.4f}'
            })
    accuracy = 100 * total_correct / total_samples
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss
