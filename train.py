import torch
from torch import nn
from myLenet_net import MyLetNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

#数据转化为Tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
    ])

#加载训练数据集，现成的
train_dataset = datasets.MNIST(root='./data',train=True, transform=data_transform, download=True)
train_dataLoader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 16,shuffle = True)

#加载测试数据集
test_dataset = datasets.MNIST(root='./data',train=False, transform=data_transform, download=True)
test_dataLoader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 16,shuffle = True)

#显卡GPU训练
device = "cuda" if torch.cuda.is_available() else "cpu"

#调用网络模型，将模型数据转到GPU
model = MyLetNet5().to(device)

#定义一个损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

#定义优化器
optimistic = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9)

#学习率每隔十轮变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimistic,step_size=10,gamma=0.1)


#定义训练函数
def train (dataloader,model,loss_fn,optimistic):
    loss , current, n = 0.0,0.0,0.0
    for batch, (X,y) in enumerate (dataloader):
        #前向传播
        X,y = X.to(device),y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y )
        _ ,pred = torch.max(output,axis = 1)
        cur_acc = torch.sum(y == pred)/output.shape[0]

        #反向传播
        optimistic.zero_grad()
        cur_loss.backward()
        optimistic.step()

        #计算值
        loss += cur_loss.item()
        current += cur_acc.item()

        n+=1
    print("loss："+ str(loss/n))
    print("accuracy："+ str(current/n))

def val(dataloader, model,loss_fn):
    model.eval()
    loss , current, n = 0.0,0.0,0.0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            # 计算值
            loss += cur_loss.item()
            current += cur_acc.item()

            n += 1
        print("val_loss：" + str(loss / n))
        print("val_accuracy：" + str(current / n))
    return current/n

#开始训练
epoch = 50
min_acc = 0
for i in range (epoch):
    print(f'epoch{i+1}----------')
    train(train_dataLoader,model,loss_fn,optimistic)
    a = val(test_dataLoader ,model,loss_fn)
    #保存最好的精确度模型权重
    if a>min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = a
        print("save best model")
        torch.save(model.state_dict(),'save_model/best_model.pth')
print('done!!!!')


