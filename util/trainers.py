import math

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from util.gradients import compute_gradient_cosine, decompose_gradient
from util.other_setting import init
from util.tester import test_acc


def pretrain_finetune(model, src_train_loader, tar_train_loader, tar_test_loader, criterion, optimizer,
                      device, num_epochs):
    init()
    scaler = torch.amp.GradScaler('cuda')
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

            with torch.amp.autocast('cuda'):
                optimizer.zero_grad(set_to_none=True)
                y_hat = model(x)
                loss = criterion(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss / len(pbar):.4f}'})

        test_accuracy, test_loss = test_acc(model, tar_test_loader, criterion)
        test_accuracies.append(test_accuracy)

    return test_accuracies


def train_single(model, train_loader, test_loader, criterion, optimizer, device, num_epochs):
    init()
    scaler = torch.amp.GradScaler('cuda')
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
                feature, y_hat = model(x)
                loss = criterion(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss / len(pbar):.4f}'})

        test_accuracy, test_loss = test_acc(model, test_loader, criterion, device)
        test_accuracies.append(test_accuracy)
    return test_accuracies


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
