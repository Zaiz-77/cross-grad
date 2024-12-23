import os
from datetime import datetime

from util.other_setting import init
from util.outputs import save_config, save_outs
from util.trainers import pretrain_finetune, train_single, train_joint


def one_exp(model, src_train, tar_train, cls_loss, optimizer, device, num_epochs, tar_test, mode):
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
        'cls_loss': cls_loss.__class__.__name__,
        'optimizer': optimizer.__class__.__name__,
        'lr': optimizer.param_groups[0]['lr'],
        'device': str(device),
        'num_epochs': num_epochs,
        'tar_test': tar_test.domain
    }
    save_config(config, out_dir)

    if mode == 'pre':
        acc = pretrain_finetune(model, src_train, tar_train, tar_test, cls_loss, optimizer, device, num_epochs)
        save_outs(acc, num_epochs, 'pretrain_finetune', out_dir)

    elif mode == 't2t':
        acc = train_single(model, tar_train, tar_test, cls_loss, optimizer, device, num_epochs)
        save_outs(acc, num_epochs, 'single_t2t', out_dir)

    elif mode == 's2t':
        acc = train_single(model, src_train, tar_test, cls_loss, optimizer, device, num_epochs)
        save_outs(acc, num_epochs, 'single_s2t', out_dir)

    elif mode == 'joint':
        acc = train_joint(model, src_train, tar_train, cls_loss, optimizer, device, num_epochs, tar_test)
        save_outs(acc, num_epochs, 'joint_st2t', out_dir)



