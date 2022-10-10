import timm
import torch

from .model import Model
from .registry import register_model

class ViT(Model):
    def __init__(self, args, model, ckp):
        super().__init__(args, model, ckp)
        print("torch version:", torch.__version__)
        print("timm version:", timm.__version__)
        self.model = self.get_model(model, ckp)
        self.use_avg = args.avg
        print("use_avg:", self.use_avg)
        print(model)
        print(self.transform)

    def get_model(self, model, ckp):
        return timm.create_model(model_name=model, pretrained=True)

    def postprocessing(self, feature):
        if timm.__version__ >= '0.6.5':
            # cls [batch, 197, 768] -> [batch, 768]
            if not self.use_avg:
                feature = feature[:, 0]
            else:
                feature = feature.transpose(1, 2)
                feature = torch.nn.AdaptiveAvgPool1d(1)(feature)
                feature = feature.squeeze(2)
        return feature

@register_model
def vit(args, model, ckp):
    return ViT(args, model, ckp)