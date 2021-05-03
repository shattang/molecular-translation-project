import torch.nn as nn
import torchvision.models as models

# starter code from https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/execution


class EncoderCNN(nn.Module):
    def __init__(self, resnet_model=None, fine_tune=False):
        super().__init__()
        if not resnet_model is None:
            resnet = resnet_model
        else:
            resnet = models.resnet18(pretrained=True)
        self._create_conv_layer(resnet, fine_tune)

    def forward(self, images):
        features = self.conv_layer(images)  # e.g. resnet18: (N, 512, 7, 7)
        features = features.permute(0, 2, 3, 1)  # (N, 7, 7, 512)
        features = features.view(features.size(
            0), -1, features.size(-1))  # (N, 49, 512)
        return features

    def _create_conv_layer(self, resnet, fine_tune):
        modules = list(resnet.children())[:-2]
        self.conv_layer = nn.Sequential(*modules)
        for param in self.conv_layer.parameters():
            param.requires_grad = False
        for child in list(self.conv_layer.children())[5:]:
            for param in child.parameters():
                param.requires_grad = fine_tune
