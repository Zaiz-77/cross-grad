import os

import yaml
from matplotlib import pyplot as plt


def save_outs(acc, num_epochs, cross, save_dir):
    best = 0
    pos = -1
    for i, ac in enumerate(acc):
        if ac > best:
            best = ac
            pos = i
    out_path = os.path.join(save_dir, f'best.txt')
    with open(out_path, 'w') as f:
        f.write(f'Best Accuracy {best}% at epoch {pos + 1}\n')

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
