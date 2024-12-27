import math

import torch
from torchtoolbox.nn import FocalLoss
from tqdm import tqdm

from model.office_model import OfficeModel
from planb.generate_pseudo_labels import pseudo_label_loader
from specific_loader import domain_loader
from util.gradients import compute_gradient_cosine, decompose_gradient
from util.tester import test_acc


def train_joint(model, src_train, tar_train, cls_loss, optimizer, device, num_epochs, tar_test):
    scaler = torch.amp.GradScaler('cuda')
    test_accuracies = []

    alpha_0 = 0.7
    alpha_min = 0.06

    T_max = 32

    for epoch in range(num_epochs):
        cnt = 0
        model.train()
        src_iter = iter(src_train)
        tar_iter = iter(tar_train)
        step_per_epoch = len(src_train)

        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / T_max))
        alpha = alpha_min + (alpha_0 - alpha_min) * cosine_decay
        alpha = max(0.0, alpha)

        p_bar = tqdm(total=step_per_epoch, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for step in range(step_per_epoch):
            try:
                src_x, src_y = next(src_iter)
            except StopIteration:
                src_iter = iter(src_train)
                src_x, src_y = next(src_iter)

            try:
                tar_x, tar_y = next(tar_iter)
            except StopIteration:
                tar_iter = iter(tar_train)
                tar_x, tar_y = next(tar_iter)

            src_x = src_x.to(device, non_blocking=True)
            src_y = src_y.to(device, non_blocking=True)
            tar_x = tar_x.to(device, non_blocking=True)
            tar_y = tar_y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                optimizer.zero_grad(set_to_none=True)
                src_features, src_outs = model(src_x)
                src_loss = cls_loss(src_outs, src_y)

                scaler.scale(src_loss).backward(retain_graph=True)
                src_cls_grad = [p.grad.detach().clone() for p in model.parameters()]

                optimizer.zero_grad(set_to_none=True)
                tar_features, tar_outs = model(tar_x)
                tar_loss = cls_loss(tar_outs, tar_y)

                scaler.scale(tar_loss).backward()
                tar_cls_grad = [p.grad.detach().clone() for p in model.parameters()]

                cos_value = compute_gradient_cosine(src_cls_grad, tar_cls_grad)
                if cos_value > 0:
                    final_grads = [alpha * g1 + g2 for g1, g2 in zip(src_cls_grad, tar_cls_grad)]
                else:
                    cnt += 1
                    _, v = decompose_gradient(src_cls_grad, tar_cls_grad)
                    final_grads = [alpha * g1 + g2 for g1, g2 in zip(v, tar_cls_grad)]

                for param, grad in zip(model.parameters(), final_grads):
                    param.grad = grad

                scaler.step(optimizer)
                scaler.update()

            p_bar.set_postfix({
                'ratio': f'{100 * cnt / step_per_epoch:.2f}%',
                'src_loss': f'{src_loss.item():.4f}',
                'tar_loss': f'{tar_loss.item():.4f}',
                'alpha': f'{alpha:.4f}'
            })
            p_bar.update()

        p_bar.close()
        test_accuracy, test_loss = test_acc(model, tar_test, cls_loss, device)
        test_accuracies.append(test_accuracy)
    return test_accuracies


src, tar = 'Art', 'Product'
src_train, src_test = domain_loader(src, batch_size=16, use_transforms=True)
_, tar_test = domain_loader(tar, batch_size=16, use_transforms=True)
tar_train = pseudo_label_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OfficeModel(65).to(device)
criterion = FocalLoss(classes=65, gamma=2)
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-3)

train_joint(model, src_train, tar_train, criterion, optimizer, device, 100, tar_test)
