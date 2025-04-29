import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

loss_function =nn.CrossEntropyLoss() ##TODO: put your loss function here


class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout_prob):
        super(Net,self).__init__()
        assert len(hidden_dim)==3, "hidden_dims list must contain three elements"

        self.dropout=nn.Dropout(dropout_prob)
        
        self.lfc1=nn.Linear(input_dim,hidden_dim[0])
        self.lfc2=nn.Linear(hidden_dim[0],hidden_dim[1])
        self.lfc3=nn.Linear(hidden_dim[1],hidden_dim[2])
        self.lfc4=nn.Linear(hidden_dim[2],output_dim)

    def forward(self,x):
        x=x.view(x.size(0),-1) # size是方法

        x=F.relu(self.lfc1(x))
        x=self.dropout(x)
       
        x=F.relu(self.lfc2(x))
        x=self.dropout(x)

        x=F.relu(self.lfc3(x))

        x=F.relu(self.lfc4(x))

        return x
    

# Define the number of folds and batch size
def train_and_validate(trainset, k_folds, num_epochs,batch_size, device):
    #k_folds = 5
    #batch_size = 40
    #num_epochs=40  #this might take a while...

    results = {}  # For fold results of validation 


    loss_function =nn.CrossEntropyLoss() ##TODO: put your loss function here

    kfold = KFold(n_splits=k_folds, shuffle=True)# TODO:Initialize the k-fold cross validation using KFold

    #fig = plt.figure(figsize=(8, 6))
        # 交叉验证分组
    for fold, (train_ids, val_ids) in enumerate(kfold.split(trainset)):  #LOOP over the different folds


        print(f'FOLD {fold+1}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        #TODO: Define data loaders for training and testing data in this fold using torch.utils.data.DataLoader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler=train_subsampler, # 这里用了一个SubsetRandomSampler来指定如何抽取样本时，就不应该再设置shuffle，因为SubsetRandomSampler本身就是以随机的方式从数据集中抽取样本。
                                            num_workers=2)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=val_subsampler,
                                            num_workers=2)
        # 开始建立模型对象
        network = Net(input_dim=3*32*32, hidden_dim=[512,256,128],output_dim=10,dropout_prob=0.5).to(device)
        # 此时模型已经建立好，但是还没导入数据，之后再用直接导入数据就好

        # 训练模式
        network.train()#######写上哦

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        losses=[] # 初始化损失集合



        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):

            print(f'Starting epoch {epoch+1}')
            current_loss = 0.0 # 每次迭代初始化损失值



            # Iterate over the DataLoader for training data
            for i, (images,labels) in enumerate(trainloader, 0):# 累加损失并每处理 500 个批次输出一次平均损失。0: 这是enumerate函数的一个可选参数，它指定计数应该从0开始。
               # print(i)

            # TODO: Implement the training loop
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = network(images)
                loss = loss_function(outputs, labels)
                # losses.append(current_loss.item()) 根据后面，loss的输出知识, 它要的是每个500个batch的平均值，而这里得到的是一个Batch（ batch_size个图像同时被处理，每个图像时一行的一维数据）的loss
                loss.backward()
                optimizer.step()

                # Print statistics
                current_loss += loss.item() # 累加，没有增加元素数目
                if i % 500 == 499: #。具体来说，它是在一个训练循环的迭代中，用于每处理完特定数量的小批量（mini-batches）数据后，输出当前累积的平均损失并重置累积损失计数。
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))

                    losses.append(current_loss/500) # epoch可以比500小
                    current_loss = 0.0
        
        plt.plot([(i+1) for i in range(len(losses))], losses)# i+1从1到625， loss记录每个epoch的前500个batch的平均值
        plt.legend()
        plt.xlabel('Number of iterations') ####横坐标是迭代epoch
        plt.ylabel('Loss / Cost')
        plt.title(f'fold {fold+1}')
        plt.show()


        print('Training process has finished. Saving trained model.')

        print('Starting validation')

        # Saving the model
        save_path = f'./model-fold-{fold+1}.pth'
        #TODO: save the model using the function  torch.save
        torch.save(network.state_dict(), save_path)


        # Evaluation for this fold
        network.eval()
        correct, total = 0, 0    

        with torch.no_grad():
            

        # Iterate over the validation data and generate predictions
            for i, data in enumerate(valloader, 0):
            # TODO: Get model predictions on the validation set
                inputs, targets = data #####
                inputs,targets=inputs.to(device), targets.to(device)
                outputs=network(inputs)
                _,predicted=torch.max(outputs,1) # 第一个值（在这里用_接收，表示我们对它不感兴趣）是每个图像最高预测概率的实际值。

                total += targets.size(0) # 这个fold中评估集的包含的总图数目，评估集包含很多batch
                correct += (predicted == targets).sum().item()
        network.train()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    # 计算所有折的平均性能
    sum = 0.0
    #plt.legend(['Fold0=1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']) 
    for key, value in results.items():
        print(f'Fold {key+1}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %') # 两个fold的平均validation accuracy # tensor to device

    return results


# Test results on ALL folds:
def test(testloader, save_path,device):


    total=0
    correct=0
   
    network =Net(input_dim=3*32*32, hidden_dim=[512,256,128],output_dim=10,dropout_prob=0.5).to(device)

    # TODO: load model of the fold F using network.load_state_dict:
    network.load_state_dict(torch.load(save_path))
    network.eval()
    with torch.no_grad():


        for i, data in enumerate(testloader, 0): ## 跟前面比，用新的数据集testloader来测试的

            inputs, targets = data  # Get inputs
            # TODO: Get model predictions on the test set
            inputs, targets = data #####
            inputs,targets=inputs.to(device), targets.to(device)
            outputs=network(inputs)
            _,predicted=torch.max(outputs,1) # 第一个值（在这里用_接收，表示我们对它不感兴趣）是每个图像最高预测概率的实际值。

            total += targets.size(0) # 这个fold中评估集的包含的总图数目，评估集包含很多batch
            correct += (predicted == targets).sum().item()

        # Print accuracy
        accuracy=100.0 * correct / total
        print(f'Accuracy of the network on the test images: {accuracy}%')

        return accuracy


if __name__=='__main__':
   
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device (CPU or GPU)


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])###################

    batch_size=40

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    fig = plt.figure(figsize=(8, 6))
            # 交叉验证分组
    results = train_and_validate(trainset, k_folds=5, num_epochs=40, batch_size=40, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print('-------------------------------------------------------------------------------------------------------------')
    accuracies=[]# list is not callable
    k_folds=5
    for fold in range(k_folds):
        print(f'Testing fold {fold+1}')
        save_path=f'./model-fold-{fold+1}.pth'
        accuracy=test(testloader,save_path,device)
        accuracies.append(accuracy) # {} dict object has not attribute "append"
    sum=0
    for key, value in enumerate(accuracies):
        print(f'Fold {key+1}: {value} %')
        sum += value
    print(f'Average: {sum/len(accuracies)} %') 









