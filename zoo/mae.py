import timm
import torch

from .model import Model
from .registry import register_model
from .mae_fb import models_mae

class MAE(Model):
    def __init__(self, model, ckp):
        super().__init__(model, ckp)
        print("torch version:", torch.__version__)
        print("timm version:", timm.__version__)
        self.model = self.get_model(model, ckp)
        print(model)
        print(self.transform)

    def get_model(self, model, ckp):
        # build model
        model = getattr(models_mae, model)()
        # load model
        checkpoint = torch.load(ckp, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        return model

    def postprocessing(self, feature):
        feature = feature[:, 0]
        return feature

@register_model
def mae(model, ckp):
    return MAE(model, ckp)