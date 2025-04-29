
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm




# Load the dataset of the MR images and the corresponding ground truth tumor segmentations. We consider 2D slices of the brain.
total_train_slices = torch.load('./total_train_slices.pt')
total_train_labels = torch.load('./total_train_labels.pt')
total_train_labels[total_train_labels > 0 ] = 1  #All tumor classes of the BRATS dataset are merged into 1. Thus, we have a binary classification between "tumor" and "not tumor"

total_val_slices = torch.load('./total_val_slices.pt')
total_val_labels = torch.load('./total_val_labels.pt')
total_val_labels[total_val_labels > 0 ] = 1   #All tumor classes of the BRATS dataset are merged into 1. Thus, we have a binary classification between "tumor" and "not tumor"


# device = torch.device("cpu")  #Move to the GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Move to the GPU if available, otherwise stay on CPU


n_epochs = 30  #define the number of training epochs
batch_size = 32
epoch_loss_list = []

#class UNet(nn.Module):
#import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.conv1(torch.cat([x4, x], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x3, x], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x2, x], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x1, x], dim=1))
        logits = self.outc(x)

        return logits


  #Your implementation of the UNet goes here. You can also check out Github for ideas

model = UNet(n_channels=1, n_classes=1)


loss_function=nn.BCEWithLogitsLoss()  #choose a suitable loss function

model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)   #define the Adam optimizer


train = True

if train is True:

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        indexes = list(torch.randperm(total_train_slices.shape[0]))  # shuffle training data new
        data_train = total_train_slices[indexes]  # shuffle the training data
        labels_train = total_train_labels[indexes]
        subset_2D = zip(data_train.split(batch_size), labels_train.split(batch_size))
        progress_bar = tqdm(enumerate(subset_2D), total=len(indexes) / batch_size)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, (a, b) in progress_bar:
            images = Variable(a, requires_grad=True).to(device)  #this is the input MR image
            seg = b.to(device)  # this is the ground truth segmentation
            optimizer.zero_grad(set_to_none=True)
            output=model(images)  #this is the predicted segmentation
            loss = loss_function(output, seg)  #compare the prediction to the ground truth segmentation
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))


    plt.title("Learning Curves U-Net Model", fontsize=20)  #Plot the loss curve
    plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
    torch.save(model.state_dict(), './segmodel.pt')   #save the trained model


else:
    model.load_state_dict(torch.load('./segmodel.pt'))   #load the pretrained model
    print('loaded model')

def dice_coeff(im1, im2, empty_score=1.0):  #this function computes the dice coefficient between two binary images.
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum



inputimg = total_val_slices[10]  # Pick an input slice of the validation set to be segmented
inputlabel = total_val_labels[10] # Check out the ground truth label mask

plt.figure("input")   #Plot the input image of the validation set
plt.imshow(inputimg[0,...], vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure("ground truth segmentation")  #Plot the ground truth segmentation
plt.imshow(inputlabel[0,...], vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()


input_img = inputimg[None, ...].to(device)
prediction=model(input_img)
prediction = torch.where(prediction > 0.5, 1, 0).float()  # a binary mask is obtained via thresholding

plt.figure("prediction")   #Plot the model prediction
plt.imshow(prediction[0,0,...].cpu(), vmin=0, vmax=1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()

score = dice_coeff(prediction[0, 0].cpu(), inputlabel.cpu())  #compute the dice score
print("Dice score of the example" , score)

