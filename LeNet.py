import os
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


#要多线程运行？觉得速度略慢，应该如何修改

batch_size = 50
num_epochs = 20

#读取数据
#compose:使用多个transforms
transformation = transform = transforms.Compose([
        #将图片的像素点范围由[0, 255]转化为[0.0, 1.0],并且变为tensor
        transforms.ToTensor(),
        #([0, 1] - 0.5) / 0.5 = [-1, 1]，将像素点范围更改为[-1, 1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

#确定数据集的来源在这个模块，要是用自己的数据集该如何修改？
#transforms模块中的Compose( )把多个变换组合在一起
#下载cifar数据集
train_dataset = datasets.CIFAR10('D:\Deep_Learning\Pytorch\LeNet5\cifar-10-batches-py', train=True, transform=transformation,
                               download=True)
test_dataset = datasets.CIFAR10('D:\Deep_Learning\Pytorch\LeNet5\cifar-10-batches-py', train=False, transform=transformation,
                               download=True)

#载入并打乱数据
#train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
#加载测试集
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
#将原本的训练集分出1/5为验证集
train_db, val_db = torch.utils.data.random_split(train_dataset, [40000, 10000])
#训练集
train_loader = torch.utils.data.DataLoader(train_db,batch_size=batch_size,shuffle=True)
# 验证集
val_loader = torch.utils.data.DataLoader(val_db,batch_size=batch_size,shuffle=True)


#网络定义
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = torch.nn.Sequential(
            #卷积层1+最大池化层1
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #卷积层2+最大池化层2
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #输出尺寸为5X5X16
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        #相当于reshape,变矩阵的格式  输出的格式为：16X25   以这个格式输入dense
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x


net = Net().cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.0005,)


def net_train(net, train_data_load, optimizer, epoch, val_loader):
    net.train()
    # 样本总数
    total = len(train_data_load.dataset)
    # 样本批次训练的损失函数值的和
    train_loss = 0
    #判断正确的样本数
    num_ture = 0

    #使用这个函数标记这是第几(i)张图，0是设置起始位置
    for i, data in enumerate(train_data_load, 0):
        img, label = data
            #GPU加速
        img, label = img.cuda(), label.cuda()
        #把梯度置零，也就是把loss关于weight的导数变成0
        optimizer.zero_grad()
        preds = net(img)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        #累加损失值    item函数返回遍历的数组
        train_loss += loss.item()
        #  _,是什么意思？ 不是返回值
        _, predicted = torch.max(preds.data, 1)
        num_ture += (predicted == label).sum()

        if(i +1) % 100 ==0:
            #损失均值
            loss_mean = train_loss / (i + 1)
            #已经训练的样本数
            traind_total = (i + 1) * len(label)
            acc = 100. * num_ture / traind_total
            progress = 100. * traind_total / total
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Train_Acc: {:.6f}'.format(
                    epoch, traind_total, total, progress, loss_mean, acc))


        #validation
def Val():
        with torch.no_grad():
            num_ture_v = 0
            for j, data in enumerate(val_loader, 0):
                img_v, label_v = data
                # GPU加速
                img_v, label_v = img_v.cuda(), label_v.cuda()
                outs = net(img_v)
                _, pre = torch.max(outs.data, 1)
                num_ture_v += (pre == label_v).sum()
            acc = num_ture_v.item() * 100. / (len(val_loader.dataset))
            print('EPOCH:{}, VAL_ACC:{}\n'.format(epoch, acc))


#测试集
def test(net, test_data_load,):
    #测试模型时在前面使用
    net.eval()
    num_ture = 0
    for i, data in enumerate(test_data_load):
        img, label = data
        img, label = img.cuda(), label.cuda()
        outs = net(img)
        _, pre = torch.max(outs.data, 1)
        num_ture += (pre == label).sum()
    acc = num_ture.item() * 100. / (len(test_data_load.dataset))
    print('TEST_ACC:{}\n'.format(acc))




print(net)

#本来把epoch循环放在train函数里面，结果发现会出错，于是拿出来了，再在后面补上验证集
for epoch in range(1,num_epochs+1):
    net_train(net, train_loader, optimizer, epoch, val_loader)
    Val()
test(net, test_loader,)





