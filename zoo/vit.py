import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch

from .model import Model
from .registry import register_model

class ViT(Model):
    def __init__(self, model, ckp):
        super().__init__(model, ckp)
        print("torch version:", torch.__version__)
        print("timm version:", timm.__version__)
        self.model = self.get_model(model, ckp)
        config = resolve_data_config({}, model=model)
        self.transform = create_transform(**config)
        # print(self.model)
        # print(self.transform)

    def get_model(self, model, ckp):
        return timm.create_model(model_name=model, pretrained=True)   

    def postprocessing(self, feature):
        if timm.__version__ >= '0.6.5':
            # cls [batch, 197, 768] -> [batch, 768]
            feature = feature[:, 0]
        return feature

@register_model
def vit(model, ckp, **kwargs):
    return ViT(model, ckp)