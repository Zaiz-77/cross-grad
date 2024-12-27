import itertools

import torch
from torch import nn

from datasets.loader import get_office_home_loaders
from model.digit_model import SimpleDigitModel
from model.model_train import one_exp
from model.office_model import OfficeModel

models = {
    'digit': SimpleDigitModel(),
    'office31': OfficeModel(31),
    'office_home': OfficeModel(65)
}

if __name__ == '__main__':
    o_31 = ['amazon', 'dslr', 'webcam']
    o_home = ['Art', 'Clipart', 'Product', 'RealWorld']
    train, test = get_office_home_loaders(batch_size=16)
    # train, test = get_office31_loaders(batch_size=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100

    for src, tar in itertools.permutations(o_home, 2):
        model = OfficeModel(65).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-3)
        cls_loss = nn.CrossEntropyLoss()
        one_exp(model, train[src], train[tar], cls_loss, optimizer, device, num_epochs, test[tar], 'joint')
