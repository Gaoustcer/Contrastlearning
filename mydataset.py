from torch.utils.data import Dataset

class pairdataset(Dataset):
    def __init__(self,images,labels) -> None:
        super(pairdataset,self).__init__()
        self.images = images
        self.labels = labels
        self.numberimages = len(self.labels)
    
    def __len__(self):
        return self.numberimages ** 2
    
    def __getitem__(self,index):
        j = index % self.numberimages
        i = (index - j) // self.numberimages
        # print("i,j",i,j)
        return (self.images[i],self.images[j]),\
            torch.stack((self.labels[i],self.labels[j]))

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    images = torch.rand(128,1,28,28)
    labels = torch.rand(128)
    pair = pairdataset(images,labels)
    loader = DataLoader(pair,batch_size=32)
    for imagepair,labelpair in loader:
        # print(len(imagepair))
        print(imagepair[0].shape)
        print(labelpair.shape)
        exit()
    # print(pair[0][0][0].shape)
    print(pair[3][1])
    print(len(pair))