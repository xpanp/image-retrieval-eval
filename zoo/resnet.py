import timm
import torch

from .model import Model
from .registry import register_model

class ResNet(Model):
    def __init__(self, args, model, ckp):
        super().__init__(args, model, ckp)
        print("torch version:", torch.__version__)
        print("timm version:", timm.__version__)
        self.model = self.get_model(model, ckp)
        # 最后一层特征图做平均池化
        self.avg = torch.nn.AvgPool2d(args.avg_size, stride=1)
        print(model)
        print(self.transform)

    def get_model(self, model, ckp):
        return timm.create_model(model_name=model, pretrained=True)

    def postprocessing(self, feature):
        feature = self.avg(feature)
        feature = feature.view(1, -1)
        return feature

@register_model
def resnet(args, model, ckp):
    return ResNet(args, model, ckp)