from .registry import register_dataset
from .dataset import Dataset
from pathlib import Path
import torch
import os

# 输入文件夹路径，返回文件列表
# paris文件夹多一层文件夹
def dir_2_filelist(dir):
    names = os.listdir(dir)
    paths = []  # 所有图片路径
    for name in names:
        files = os.listdir(os.path.join(dir, name))
        for file in files:
            paths.append(os.path.join(dir, name, file))
    return paths
    
class GD(Dataset):
    def __init__(self, datadir, datapth):
        super().__init__(datadir, datapth)

    def get_id(self, path):
        # 获取文件基础名,不含前后缀
        name = Path(path).name
        return name

    def get_filelist(self):
        return dir_2_filelist(self.datadir)

    def evaluate(self):
        features = torch.load(self.datapth)
        
        for i in range(len(features)):
            for j in range(len(features)):
                score = torch.cosine_similarity(features[i][1], features[j][1], dim=1)
                print(features[i][0], features[j][0], score)


@register_dataset
def guangdong(datadir, datapth):
    return GD(datadir, datapth)
