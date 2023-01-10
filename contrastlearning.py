import torch.nn as nn
import torch
from loaddata import train_dataset,test_dataset
from model import imagefeature
from torch.utils.data import DataLoader
from mydataset import pairdataset
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt


class contrastlearning(object):
    def __init__(self) -> None:
        self.net = imagefeature().cuda()
        self.optimizer = Adam(self.net.parameters(),lr = 0.0001) 
        self.trainloader = DataLoader(train_dataset,batch_size=16)
        self.testloader = DataLoader(test_dataset,batch_size=16)
        self.EPOCH = 32
        self.m = 2
        self.colors = plt.get_cmap("RdBu",10)
        # generate color array


    def train(self):
        for epoch in range(self.EPOCH):
            self.validate("./image/image{}.png".format(epoch))
            for images,labels in tqdm(self.trainloader):
                pair = DataLoader(pairdataset(images.cuda(),labels.cuda()),batch_size=64)
                for imagespairs,labelspairs in pair:
                    self.optimizer.zero_grad()
                    Y = (labelspairs[0] == labelspairs[1])
                    positive = self.net(imagespairs[0])
                    negative = self.net(imagespairs[1])
                    distance = torch.norm(positive - negative,p=2,dim=-1)
                    loss = (Y * torch.pow(distance,2)) + (~Y * torch.pow(self.m - distance,2))
                    loss = torch.mean(loss)
                    loss.backward()
                    self.optimizer.step()
                
        
    def validate(self,path):
        for images,labels in self.testloader:
            imagesembedding = self.net(images.cuda()).cpu().detach().numpy()
            print(imagesembedding)
            colors = self.colors(labels.numpy())
            # imagesembedding
            plt.scatter(x = imagesembedding[0,:], y = imagesembedding[0:1],s = 0.1,c = colors)

        # plt.savefig("./clusterresult.png")
        plt.savefig(path)
        plt.close()
            

def _getcolors(colornum = 10):
    colors = plt.get_cmap("RdBu",colornum)
    # colorlist = []
    # for i in range(colornum):
    #     colorlist.append(colors([0]))

if __name__ == "__main__":
    # pass




    

