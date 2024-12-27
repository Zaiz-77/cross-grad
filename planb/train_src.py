import torch
from torchtoolbox.nn import FocalLoss

from model.office_model import OfficeModel
from planb.specific_loader import domain_loader


def train_source_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 在验证集上评估模型
        accuracy = evaluate_model(model, test_loader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model_on_src.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    print(f'Best Accuracy: {best_accuracy:.4f}')


# 评估模型
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


src_train, src_test = domain_loader('Art', batch_size=16, use_transforms=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OfficeModel(65).to(device)
criterion = FocalLoss(classes=65, gamma=2)
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-3)
# 开始训练
train_source_model(model, src_train, src_test, criterion, optimizer, num_epochs=100)
