import torch
from myLenet_net import MyLetNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

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
#载入已经训练好的最佳模型
model.load_state_dict(torch.load("D:/pytorch_project/project1/save_model/best_model.pth"))

#获取结果
classes = ["0","1","2","3","4","5","6","7","8","9"]

#把Tensor转化为图片
show = ToPILImage()

for i in range(5):
    X ,y = test_dataset[i][0], test_dataset[i][1]
    #展示图片
    show(X).show()

    #张量扩张为4维
    X = Variable(torch.unsqueeze(X,dim=0).float() , requires_grad = False).to(device)
    with torch.no_grad():
        pred = model(X)
        predict ,actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predict ：{predict}\tactual：{actual}')

