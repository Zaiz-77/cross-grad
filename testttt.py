from datasets.loader import get_mnist_dataloader, get_usps_dataloader, get_office31_loaders

mnist_train_loader, mnist_test_loader = get_mnist_dataloader()
usps_train_loader, usps_test_loader = get_usps_dataloader()
office31_train, office31_test = get_office31_loaders()
print(mnist_train_loader.dataset.__class__.__name__)
print(usps_train_loader.dataset.__class__)
print(office31_train['amazon'].dataset.__class__.__name__)