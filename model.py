import torch.nn as nn
from basic_model.googlenet import googlenet


def get_model(device, dim, pretrained=True):
    model = Model(dim, pretrained=pretrained)
    # model = nn.DataParallel(model)
    model = model.to(device)
    return model


class Model(nn.Module):
    def __init__(self, dim, pretrained=True):
        super(Model, self).__init__()
        self.dim = dim
        self.google_net = googlenet(pretrained=pretrained)
        self.linear = nn.Linear(1000, dim)

    def forward(self, x):
        if self.google_net.training and self.google_net.aux_logits:
            _, _, x = self.google_net(x)
        else:
            x = self.google_net(x)
        x = self.linear(x)
        norm = x.norm(dim=1, p=2, keepdim=True)
        x = x.div(norm)
        return x