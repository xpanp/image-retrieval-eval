from .registry import register_dataset
from .dataset import Dataset
from pathlib import Path
import torch
import numpy as np

class Ukbench(Dataset):
    def __init__(self, datadir, datapth):
        super().__init__(datadir, datapth)

    def get_id(self, path):
        name = Path(path).name
        names = name.split(".")
        pre_name = names[0]
        str_num = pre_name[-5:]
        num = int(str_num)
        return num

    def evaluate(self):
        features = torch.load(self.datapth)
        hits = []
        for i in range(len(features)):
            results = []
            for j in range(len(features)):
                score = torch.cosine_similarity(features[i][1], features[j][1], dim=1)
                results.append(score)
            res_tensor = torch.tensor(results)
            _, index = torch.topk(res_tensor, 4)
            hit = 0
            for w in range(4):
                if (features[index[w]][0] // 4) == (features[i][0] // 4):
                    hit += 1
            hits.append(hit)
            print(i, "/", len(features), ", num:", features[i][0], ", hit:", hit, "top4:", 
                features[index[0]][0], results[index[0]], features[index[1]][0], results[index[1]], 
                features[index[2]][0], results[index[2]], features[index[3]][0], results[index[3]])
        
        print(np.mean(hits))

@register_dataset
def ukbench(datadir, datapth):
    return Ukbench(datadir, datapth)
