import torch
import torchvision as tv
import torch as th
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score


class Flatten(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out


class ClassificationNetCNN(th.nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self._model = th.nn.Sequential(
            th.nn.Conv2d(1, 32, kernel_size=kernel_size, padding=1),
            th.nn.ReLU(),
            th.nn.MaxPool2d(2),
            th.nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1),
            th.nn.ReLU(),
            th.nn.MaxPool2d(2),
            Flatten(),
            th.nn.Linear(64 * 8 * 8, 128),
            th.nn.ReLU(),
            th.nn.Dropout(0.5),
            th.nn.Linear(128, 10)
        )

    def forward(self, input):
        # TODO: Implement your forward pass here
        logits = self._model(input)
        return logits


def eval_model(model, dataloader_test, device=torch.device('cpu')):
    model.eval()
    target_labels = []
    model_labels = []
    with torch.no_grad():
        for i, (image, target) in enumerate(dataloader_test):
            image = image.to(device)
            target = target.to(device)
            estimated_class_label = model(image)
            target_labels += list((target.cpu().numpy()))
            model_labels += list(th.argmax(estimated_class_label, dim=-1).cpu().numpy())
            if i > 50:
                break
    model.train()
    return accuracy_score(target_labels, model_labels)


def train_mnist(num_epoch=3, batch_size=100, seed=123456):
    use_tensorboard = False
    if use_tensorboard == False:
        """
        execute following line in a separate terminal to view the results:
        tensorboard --logdir=runs --bind_all
        and open the URL this command gives you in your browser
        """
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter()
    else:
        summary_writer = None

    th.manual_seed(seed)
    np.random.seed(seed)

    device = th.device("cpu")

    # model generation
    # TODO: Initialize your model here
    model = ClassificationNetCNN().to(device)

    # print number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of parameters model", params)

    # get MNIST data set for training
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()])

    data_set = tv.datasets.MNIST("/tmp/MNIST", train=True, transform=transform, target_transform=None, download=True)
    test_set = tv.datasets.MNIST("/tmp/MNIST/test", train=False, transform=transform, target_transform=None,
                                 download=True)
    print('len', len(data_set))  # amount of downloaded images

    # TODO: split the data_set into train and validation set, using  torch.utils.data.random_split
    train_size = int(0.8 * len(data_set))
    val_size = len(data_set) - train_size
    train_set, val_set = torch.utils.data.random_split(data_set, [train_size, val_size])

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 1}

    # define the data set and data loader for the training
    batch_size = batch_size
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 1}
    # TODO: build your dataloaders using data.DataLoader
    data_loader_training = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    data_loader_validation = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
    data_loader_test = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    # define optimizer
    learning_rate = 0.001
    # TODO: define your optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    # TODO:  define loss function
    loss = th.nn.CrossEntropyLoss()

    idx = 0
    val_acc = 0
    train_acc = 0
    acc = []
    for epoch in range(num_epoch):
        model.train()
        for image, target in data_loader_training:
            image = image.to(device)
            target = target.to(device)

            # TODO:  implement the training loop
            optimizer.zero_grad()
            outputs = model(image)
            loss_value = loss(outputs, target)
            loss_value.backward()
            optimizer.step()

            estimated_class_label = outputs.argmax(dim=-1)  # distribution -> number
            batch_accuracy = (target == estimated_class_label).float().mean()  # report the batch accuracy
            acc.append(batch_accuracy)

            # compute the accuracy of the model on the validation set, plot images
            if idx % 100 == 0:
                val_acc = eval_model(model, data_loader_validation, device=device)
                train_acc = np.mean([a.cpu().numpy() for a in acc])
                acc = []

                if summary_writer is not None:
                    summary_writer.add_scalar("val_acc", val_acc, idx)
                    summary_writer.add_scalar("train_acc", train_acc, idx)

                print(
                    f'Epoch = {epoch:>3}, index = {idx:>8}, loss = {loss_value.item():.5f}, train_acc ={train_acc:.5f}, val_acc = {val_acc:.5f}')

            if summary_writer is not None:
                summary_writer.add_scalar("loss", loss_value.item(), idx)
            idx = idx + 1

    # end of training: register parameters and losses
    if summary_writer is not None:
        summary_writer.add_hparams(
            hparam_dict=dict(num_epoch=num_epoch, lr=learning_rate, batch_size=batch_size),
            metric_dict=dict(final_loss=loss_value.item(), val_acc=val_acc, train_acc=train_acc),
            run_name='.')

    # TODO: evaluate the trained model on the test set
    test_acc = eval_model(model, data_loader_test, device)
    print('Accuracy on the test set', test_acc)


if __name__ == "__main__":
    train_mnist()
