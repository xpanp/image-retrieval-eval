from abc import ABC, abstractmethod
from PIL import Image
from torchvision import transforms, models
import torch

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

class Model(ABC):
    def __init__(self, model, ckp):
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #     transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
        ])
        self.model = None

    @abstractmethod
    def get_model(self, model, ckp):
        pass

    def postprocessing(self, feature):
        # feature = l2n(feature)
        pass

    def extract(self, dataset):
        paths = dataset.get_filelist()
        results = []
        for i, path in enumerate(paths):
            id = dataset.get_id(path)
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                feature = self.model.forward_features(img_tensor)
            feature = self.postprocessing(feature)
            results.append((id, feature))
            print(i, id, path)
        
        torch.save(results, dataset.get_pth_path())