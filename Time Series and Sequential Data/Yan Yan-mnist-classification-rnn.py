__author__      = "Florentin Bieder and Robin Sandkuehler"
__copyright__   = "Center for medical Image Analysis and Navigation, University of Basel, 2022"

import torchvision as tv
import torch
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score


class BasicRNN_Classification(torch.nn.Module):
    def __init__(self, input_size, state_size, num_class):
        super().__init__()

        self._state_size = state_size

        #TODO define the elements of the basic RNN equations we do not net the output for each time step (many to one)
        self.W_h = torch.nn.Linear(input_size,state_size)
        self.U_h = torch.nn.Linear(state_size,state_size)
    

        #TODO define the final layer that maps from the final state of the RNN to the class estimations
        self.classifier = torch.nn.Linear(state_size,num_class)

    def forward(self, input):

        # define and initialize the internal state
        state = torch.zeros(input.shape[0], self._state_size)

        for i in range(input.shape[2]):
            current_row = input[:, :, i, :].squeeze(1)
            #TODO implement the basic RNN equation for the state h_t here
            state = torch.tanh(self.W_h(current_row) + self.U_h(state))


        # compute the final classification vector
        logits = self.classifier(state)

        return logits


def eval_model(model, dataloader_test, device=torch.device('cpu')):
    model.eval()

    target_labels = []
    model_labels = []
    with torch.no_grad():
        for i, (image, target) in enumerate(dataloader_test):
            image = image.to(device)
            target = target.to(device)
            estimated_class_label = torch.softmax(model(image), dim=-1)
            target_labels += list((target.cpu().numpy()))
            model_labels += list(torch.argmax(estimated_class_label, dim=-1).cpu().numpy())
            if i > 50:
                break
    model.train()
    return accuracy_score(target_labels, model_labels)


def train_rnn_mnist(num_epoch=3, batch_size=100, seed=123456):
    use_tensorboard = True
    if use_tensorboard is True:
        """
        execute following line in a separate terminal to view the results:
        tensorboard --logdir=runs --bind_all
        and open the URL this command gives you in your browser
        """
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter()
    else:
        summary_writer = None

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    # model generatation Basic RNN
    model = BasicRNN_Classification(input_size=28, state_size=512, num_class=10)
    model.to(device)
    model.train()


    # print number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of parameters model", params)

    # get MNIST data set
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor()])

    data_set_training = tv.datasets.MNIST("/tmp/MNIST", train=True, transform=transform, target_transform=None,
                                          download=True)
    batch_size = batch_size
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 1}

    data_loader_training = data.DataLoader(data_set_training, **params)

    # define the data set and data loader for the training
    data_set_validation = tv.datasets.MNIST("/tmp/MNIST/test", train=False, transform=transform, target_transform=None,
                                            download=True)
    batch_size = batch_size
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 1}

    data_loader_validation = data.DataLoader(data_set_validation, **params)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    #TODO define loss function
    loss = torch.nn.CrossEntropyLoss() 

    idx = 0
    val_acc = 0
    train_acc = 0
    acc = []
    for epoche in range(num_epoch):
        for image, target in data_loader_training:
            image = image.to(device)
            target = target.to(device)

            estimated_class_label = model(image)

            loss_value = loss(estimated_class_label, target)

            optimizer.zero_grad()
            loss_value.backward()

            for p in model.parameters():
                if p.grad is None:
                    continue
                p.grad.data = p.grad.data.clamp(-1, 1)

            optimizer.step()

            estimated_class_label = torch.softmax(estimated_class_label, dim=-1).argmax(dim=-1)  # distribution -> number
            batch_accuracy = (target == estimated_class_label).float().mean()
            acc.append(batch_accuracy)

            if idx % 25 == 0:

                # compute the accuracy of the model on the validation set, plot images
                    val_acc = eval_model(model, data_loader_validation, device=device)
                    train_acc = np.mean([a.cpu().numpy() for a in acc])
                    acc = []

                    if summary_writer is not None:
                        summary_writer.add_scalar("val_acc", val_acc.item(), idx)
                        summary_writer.add_scalar("train_acc", train_acc, idx)
                        summary_writer.add_scalar("ce_loss", loss_value.item(), idx)
                        img = add_numbers(image, estimated_class_label, target)
                        summary_writer.add_images("img", img, global_step=idx, dataformats='NCHW')

                    print(f'Epoch = {epoche:>3}, index = {idx:>8}, loss = {loss_value.item():.5f}, train_acc = {train_acc:.5f}, val_acc = {val_acc:.5f}')


            idx = idx + 1


# hardcode numbers to paste into graphic
# (there are definitely better ways for larger images,
#  but these images are only 28 x 28, so we need a really small font)
pixel_nums = torch.ones((10, 15))
pixel_nums[0, [4,7,10]] = 0
pixel_nums[1, [0,3,6,9,12,2,5,8,11,14]] = 0
pixel_nums[2, [3,4,10,11]] = 0
pixel_nums[3, [3,4,9,10]] = 0
pixel_nums[4, [1,4,9,10,12,13]] = 0
pixel_nums[5, [4,5,9,10]] = 0
pixel_nums[6, [4,5,10]] = 0
pixel_nums[7, [3,4,6,7,9,10,12,13]] = 0
pixel_nums[8, [4,10]] = 0
pixel_nums[9, [4,9,10]] = 0

def add_numbers(batch, numbers, true_numbers=None):
    """
    adds numbers to batch of images, optionally colours
    bw images depending of
    batch:        batched tensor of input images
    numbers:      batch of predicted numbers
    true_numbers: batch of ground truth (optional)
    """
    for i in range(len(numbers)):
        batch[i, 0, 1:6, 1:4] = pixel_nums[numbers[i],:].reshape(5, 3)
    if true_numbers is not None:
        batch = batch.expand(-1, 3, -1, -1).clone()
        batch[:, 2, ...] = 0 # set blue to zero
        for i in range(len(numbers)):
            if numbers[i] == true_numbers[i]:
                batch[i, 0, ...] = 0 # set red to zero, make it green
            else:
                batch[i, 1, ...] = 0 # set green to zero, make it red
        #batch = (batch*255).to(torch.uint8)
        #batch = batch.clamp(0, 1)
    return batch


if __name__ == "__main__":
    train_rnn_mnist()
