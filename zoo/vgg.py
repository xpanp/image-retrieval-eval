import torch.nn as nn
import torch
from torchvision import models

from .model import Model
from .registry import register_model

class VGG16_model:
    def __init__(self) -> None:
        self.model = models.vgg16(pretrained=True)
        self.feature_extractor = self.model.features
    
    def get_k_layer_feature_map(self, k, x):
        with torch.no_grad():
            # feature_extractor是特征提取层，后面可以具体看一下vgg16网络
            for index, layer in enumerate(self.feature_extractor):
                # x是输入图像的张量数据，layer是该位置进行运算的卷积层，就是进行特征提取
                x = layer(x)
                # k代表想看第几层的特征图
                if k == index:
                    return x

    def forward_features(self, x):
        feature_map = self.get_k_layer_feature_map(28, x)
        # vgg16第28层为512*14*14,全局平均池化，提取512维特征
        avg_pool = nn.AvgPool2d(14, stride=1)
        feature_map_avg = avg_pool(feature_map)
        # 降维
        feature_map_avg_s = feature_map_avg.view(1, -1)
        return feature_map_avg_s

class VGG(Model):
    def __init__(self, model, ckp):
        super().__init__(model, ckp)
        print("torch version:", torch.__version__)
        self.model = self.get_model(model, ckp)
        print(model)
        print(self.transform)

    def get_model(self, model, ckp):
        return VGG16_model()


@register_model
def vgg(model, ckp):
    return VGG(model, ckp)