import timm
import torch

from .model import Model
from .registry import register_model
from .mae_fb import models_mae

class MAE(Model):
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
        # build model
        model = getattr(models_mae, model)()
        # load model
        checkpoint = torch.load(ckp, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        return model

    def postprocessing(self, feature):
        if not self.use_avg:
            feature = feature[:, 0]
        else:
            feature = feature.transpose(1, 2)
            feature = torch.nn.AdaptiveAvgPool1d(1)(feature)
            feature = feature.squeeze(2)
        return feature

@register_model
def mae(args, model, ckp):
    return MAE(args, model, ckp)