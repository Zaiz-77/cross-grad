import math
import os
import random
from datetime import datetime
from math import sqrt

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print("CUDA cache cleared.")


def compute_gradient_cosine(grad1, grad2):
    dot_product = sum((g1 * g2).sum() for g1, g2 in zip(grad1, grad2))
    norm1 = sqrt(sum((g * g).sum() for g in grad1))
    norm2 = sqrt(sum((g * g).sum() for g in grad2))
    return dot_product / (norm1 * norm2)


def decompose_gradient(grad_mnist, grad_usps):
    dot_product = sum((g1 * g2).sum() for g1, g2 in zip(grad_mnist, grad_usps))
    norm_usps_square = sum((g * g).sum() for g in grad_usps)

    scale = dot_product / norm_usps_square
    h = [g * scale for g in grad_usps]
    v = [g1 - g2 for g1, g2 in zip(grad_mnist, h)]

    return h, v


def save_outs(acc, num_epochs, cross, save_dir):
    epochs = range(1, num_epochs + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, acc, 'r-', label='Test Accuracy')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='center right')
    ax.grid(True)

    plt.title('Test Accuracy over Epochs')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{cross}.png')
    plt.savefig(save_path)
    plt.close()


def save_config(config, out_dir):
    config_path = os.path.join(out_dir, 'config.yaml')
    with open(config_path, 'w') as file:
        yaml.dump(config, file)


def pretrain_finetune(model, src_train_loader, tar_train_loader, tar_test_loader, criterion, optimizer,
                      device, num_epochs):
    init()
    scaler = GradScaler()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(src_train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast():
                optimizer.zero_grad(set_to_none=True)
                y_hat = model(x)
                loss = criterion(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss / len(pbar):.4f}'})

    scaler = GradScaler()
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(tar_train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast():
                optimizer.zero_grad(set_to_none=True)
                y_hat = model(x)
                loss = criterion(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss / len(pbar):.4f}'})

        test_accuracy, test_loss = test_acc(model, tar_test_loader, criterion, device)
        test_accuracies.append(test_accuracy)

    return test_accuracies


def train_single(model, train_loader, test_loader, criterion, optimizer, device, num_epochs):
    init()
    scaler = GradScaler()
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast():
                optimizer.zero_grad(set_to_none=True)
                y_hat = model(x)
                loss = criterion(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss / len(pbar):.4f}'})

        test_accuracy, test_loss = test_acc(model, test_loader, criterion, device)
        test_accuracies.append(test_accuracy)
    return test_accuracies


# def train_joint(model, src_train, tar_train, criterion, optimizer, device, num_epochs, tar_test):
#     scaler = GradScaler()
#     test_accuracies = []
#     for epoch in range(num_epochs):
#         cnt = 0
#         model.train()
#         src_iter = iter(src_train)
#         tar_iter = iter(tar_train)
#         step_per_epoch = len(src_train)
#
#         p_bar = tqdm(total=step_per_epoch, desc=f'Epoch {epoch + 1}/{num_epochs}')
#
#         for step in range(step_per_epoch):
#             try:
#                 src_x, src_y = next(src_iter)
#             except StopIteration:
#                 src_iter = iter(src_train)
#                 src_x, src_y = next(src_iter)
#
#             try:
#                 tar_x, tar_y = next(tar_iter)
#             except StopIteration:
#                 tar_iter = iter(tar_train)
#                 tar_x, tar_y = next(tar_iter)
#
#             src_x = src_x.to(device, non_blocking=True)
#             src_y = src_y.to(device, non_blocking=True)
#             tar_x = tar_x.to(device, non_blocking=True)
#             tar_y = tar_y.to(device, non_blocking=True)
#
#             with autocast():
#                 optimizer.zero_grad(set_to_none=True)
#                 src_outs = model(src_x)
#                 src_loss = criterion(src_outs, src_y)
#
#             scaler.scale(src_loss).backward(retain_graph=True)
#             src_grads = [p.grad.detach().clone() for p in model.parameters()]
#
#             with autocast():
#                 optimizer.zero_grad(set_to_none=True)
#                 tar_outs = model(tar_x)
#                 tar_loss = criterion(tar_outs, tar_y)
#
#             scaler.scale(tar_loss).backward()
#             tar_grads = [p.grad.detach().clone() for p in model.parameters()]
#
#             cos_value = compute_gradient_cosine(src_grads, tar_grads)
#             if cos_value > 0:
#                 final_grads = [g1 + g2 for g1, g2 in zip(src_grads, tar_grads)]
#             else:
#                 cnt += 1
#                 _, v = decompose_gradient(src_grads, tar_grads)
#                 final_grads = [g1 + g2 for g1, g2 in zip(v, tar_grads)]
#
#             for param, grad in zip(model.parameters(), final_grads):
#                 param.grad = grad
#
#             scaler.step(optimizer)
#             scaler.update()
#
#             p_bar.set_postfix({
#                 'ratio': f'{100 * cnt / step_per_epoch:.2f}%',
#                 'src_loss': f'{src_loss.item():.4f}',
#                 'tar_loss': f'{tar_loss.item():.4f}'
#             })
#             p_bar.update()
#
#         p_bar.close()
#         test_accuracy, test_loss = test_acc(model, tar_test, criterion, device)
#         test_accuracies.append(test_accuracy)
#     return test_accuracies


def train_joint(model, src_train, tar_train, criterion, optimizer, device, num_epochs, tar_test):
    init()
    scaler = GradScaler()
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

            with autocast():
                optimizer.zero_grad(set_to_none=True)
                src_outs = model(src_x)
                src_loss = criterion(src_outs, src_y)

            scaler.scale(src_loss).backward(retain_graph=True)
            src_grads = [p.grad.detach().clone() for p in model.parameters()]

            with autocast():
                optimizer.zero_grad(set_to_none=True)
                tar_outs = model(tar_x)
                tar_loss = criterion(tar_outs, tar_y)

            scaler.scale(tar_loss).backward()
            tar_grads = [p.grad.detach().clone() for p in model.parameters()]

            cos_value = compute_gradient_cosine(src_grads, tar_grads)
            if cos_value > 0:
                final_grads = [alpha * g1 + g2 for g1, g2 in zip(src_grads, tar_grads)]
            else:
                cnt += 1
                _, v = decompose_gradient(src_grads, tar_grads)
                final_grads = [alpha * g1 + g2 for g1, g2 in zip(v, tar_grads)]

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
        test_accuracy, test_loss = test_acc(model, tar_test, criterion, device)
        test_accuracies.append(test_accuracy)
    return test_accuracies


def one_exp(model, src_train, tar_train, criterion, optimizer, device, num_epochs, tar_test, mode):
    init()
    src_name = src_train.dataset.__class__.__name__
    tar_name = tar_test.dataset.__class__.__name__

    src_domain = src_train.domain
    tar_domain = tar_test.domain
    timestamp = datetime.now().strftime('%m-%d_%H-%M-%S')
    out_dir = os.path.join('out', f'{src_name}-{tar_name}', f'{timestamp}-{src_domain}->{tar_domain}')
    os.makedirs(out_dir, exist_ok=True)

    config = {
        'model': model.__class__.__name__,
        'src_train': src_train.domain,
        'tar_train': tar_train.domain,
        'criterion': criterion.__class__.__name__,
        'optimizer': optimizer.__class__.__name__,
        'lr': optimizer.param_groups[0]['lr'],
        'device': str(device),
        'num_epochs': num_epochs,
        'tar_test': tar_test.domain
    }
    save_config(config, out_dir)

    if mode == 'pre':
        acc = pretrain_finetune(model, src_train, tar_train, tar_test, criterion, optimizer, device, num_epochs)
        save_outs(acc, num_epochs, 'pretrain_finetune', out_dir)

    elif mode == 't2t':
        acc = train_single(model, tar_train, tar_test, criterion, optimizer, device, num_epochs)
        save_outs(acc, num_epochs, 'single_t2t', out_dir)

    elif mode == 's2t':
        acc = train_single(model, src_train, tar_test, criterion, optimizer, device, num_epochs)
        save_outs(acc, num_epochs, 'single_s2t', out_dir)

    elif mode == 'joint':
        acc = train_joint(model, src_train, tar_train, criterion, optimizer, device, num_epochs, tar_test)
        save_outs(acc, num_epochs, 'joint_st2t', out_dir)


def test_acc(model, dataloader, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    pbar = tqdm(dataloader, desc='Testing', ascii=True)
    with torch.no_grad(), autocast():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

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
