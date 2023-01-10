import torch.nn as nn

class imagefeature(nn.Module):
    def __init__(self) -> None:
        super(imagefeature,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=2
            ),
            nn.Flatten(),
            nn.Linear(4,2),
        )

    def forward(self,images):
        return self.net(images)