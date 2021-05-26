import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score



class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(28*28,500)
        self.fc2=nn.Linear(500,240)
        self.fc3=nn.Linear(240,120)
        self.fc4=nn.Linear(120,50)
        self.fc5=nn.Linear(50,10)
    
    def forward(self,x):
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.softmax(self.fc5(x), dim=1)
        return x


if __name__=='__main__':
    mnist_train=datasets.MNIST(root='./data',train=True,download=True,transform=ToTensor())
    mnist_train_dataloader=torch.utils.data.DataLoader(mnist_train,batch_size=1000)
    mnist_test=datasets.MNIST(root='./data',train=False,download=True,transform=ToTensor())
    mnist_test_dataloader=torch.utils.data.DataLoader(mnist_test,batch_size=1000)


    mlp=MLP()


    optimizer=torch.optim.Adam(mlp.parameters())
    for epoch in range(20):
    
        accuracy_scores=[0]*60
        for i,(X,y) in enumerate(mnist_train_dataloader):
            mlp.train()
            optimizer.zero_grad()
            out=mlp.forward(X)
            loss_fn=nn.CrossEntropyLoss()
            loss=loss_fn(out,y)
            loss.backward()
            optimizer.step()

            mlp.eval()
            y_pred=mlp.forward(X).argmax(dim=-1)
            accuracy_scores[i]=accuracy_score(y_pred,y)
        print('epoch',epoch)
        print('accuracy',sum(accuracy_scores)/60)


    mlp.eval()
    accuracy_scores=[0]*10
    for i,(X,y) in enumerate(mnist_test_dataloader):
        y_pred=mlp.forward(X).argmax(dim=-1)
        accuracy_scores[i]=accuracy_score(y_pred,y)
    print('test accuracy:')
    print(sum(accuracy_scores)/10)
    torch.save(mlp,'mlp.sav')