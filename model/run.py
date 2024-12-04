import torch
from torch import nn

from datasets.loader import get_office31_loaders, get_office_home_loaders, get_mnist_dataloader, get_usps_dataloader
from model.digit_model import SimpleDigitModel
from model.model_train import one_exp
from model.office_model import OfficeModel

models = {
    'digit': SimpleDigitModel(),
    'office31': OfficeModel(31),
    'office_home': OfficeModel(65)
}

if __name__ == '__main__':
    # mnist_train, mnist_test = get_mnist_dataloader()
    # usps_train, usps_test = get_usps_dataloader()
    # office31_train, office31_test = get_office31_loaders()
    office_home_train, office_home_test = get_office_home_loaders()
    src = 'Clipart'
    tar = 'Art'

    model = models['office_home']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    # one_exp(model, mnist_train, usps_train, criterion, optimizer, device, num_epochs, usps_test)
    # one_exp(model, office31_train[src], office31_train[tar], criterion, optimizer, device, num_epochs,
    #         office31_test[tar])
    one_exp(model, office_home_train[src], office_home_train[tar], criterion, optimizer, device, num_epochs,
            office_home_test[tar])
