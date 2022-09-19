from .registry import register_dataset
from .dataset import Dataset
from pathlib import Path
import torch
import os

def get_query(gtfile):
    querys = []
    q_v = {}
    with open(gtfile, 'r') as f:
        list = f.readlines()
        for it in list:
            it = it.strip('\n')
            qs = it.split(' ')
            if len(qs) < 7:
                querys.append(qs[0])
                q_v[qs[0]] = qs
    return querys, q_v

# 计算单个查询的ap值
def comput_ap(gt, rank):
    print(gt)
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0

    intersect_size = 0
    j = 0

    for r in rank:
        if r[0] in gt:
            intersect_size += 1
        
        recall = intersect_size / float(len(gt))
        precision = intersect_size / (j + 1.0)

        ap += (recall-old_recall) * ((old_precision+precision)/2.0)
        old_recall = recall
        old_precision = precision
        j += 1
    return ap
    
class Pattern(Dataset):
    def __init__(self, datadir, datapth):
        super().__init__(datadir, datapth)
        self.datadir = os.path.join(datadir, 'pattern')
        self.gtfile = os.path.join(datadir, 'pattern.txt')

    def get_id(self, path):
        # 获取文件基础名,不含前后缀
        name = Path(path).name
        return name

    def evaluate(self):
        features = torch.load(self.datapth)
        querys, q_v = get_query(self.gtfile)
        sum_ap = 0.0
        n = 0
        for i in range(len(features)):
            if features[i][0] in querys:
                n += 1
                results = []
                for j in range(len(features)):
                    score = torch.cosine_similarity(features[i][1], features[j][1], dim=1)
                    results.append(score)
                res_tensor = torch.tensor(results)
                _, index = torch.topk(res_tensor, len(results))
                ranked_list = []
                for t in index:
                    ranked_list.append([features[t][0], results[t]])
                ap = comput_ap(q_v[features[i][0]], ranked_list)
                print(n, "/", len(querys), ", query:", features[i][0], ", ap:", ap)
                sum_ap += ap

        print("mAP: %.5f" % (sum_ap/len(querys)))

@register_dataset
def pattern(datadir, datapth):
    return Pattern(datadir, datapth)
