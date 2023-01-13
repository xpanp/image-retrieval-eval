import timm
import torch
from PIL import Image

from .model import Model
from .registry import register_model
from .mae_fb import models_mae
from .mae import MAE
from .vit import ViT

class Fusion(Model):
    def __init__(self, args, model, ckp):
        super().__init__(args, model, ckp)
        print("torch version:", torch.__version__)
        print("timm version:", timm.__version__)
        self.model1 = MAE(args, model, ckp)
        self.model2 = ViT(args, "vit_base_patch16_224", ckp)

    def get_model(self, model, ckp):
        pass

    def postprocessing(self, feature):
        return feature

    def fusion(self, feature1, feature2):
        feature = feature1 + feature2
        print("Debug feature:", feature)
        print("Debug feature1:", feature1)
        print("Debug feature2:", feature2)
        return feature
    
    def extract(self, dataset):
        paths = dataset.get_filelist()
        results = []
        for i, path in enumerate(paths):
            id = dataset.get_id(path)
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                feature1 = self.model1.model.forward_features(img_tensor)
                feature2 = self.model2.model.forward_features(img_tensor)
            feature1 = self.model1.postprocessing(feature1)
            feature2 = self.model2.postprocessing(feature2)
            feature = self.fusion(feature1, feature2)
            # feature = l2n(feature)
            results.append((id, feature))
            print(i, id, path)
        
        torch.save(results, dataset.get_pth_path())

@register_model
def fusion(args, model, ckp):
    return Fusion(args, model, ckp)