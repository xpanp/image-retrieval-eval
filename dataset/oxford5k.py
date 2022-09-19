from .registry import register_dataset
from .dataset import Dataset
import torch
import os

def load_query(filename):
    # 如: oxc1_all_souls_000013 136.5 34.1 648.5 955.7
    # 获取: all_souls_000013
    with open(filename, 'r') as f:
        s = f.readlines()
    return s[0].split(' ')[0][5:]

def load_list(filename):
    with open(filename, 'r') as f:
        l = f.readlines()
    lists = [x.strip() for x in l]
    return lists

def get_query(query_list, dir):
    querys = []
    q_v = {}
    for i in range(len(query_list)):
        for j in range(1, 6):
            queryfile = os.path.join(dir, query_list[i] + '_' + str(j) + '_query.txt')
            goodfile = os.path.join(dir, query_list[i] + '_' + str(j) + '_good.txt')
            okfile = os.path.join(dir, query_list[i] + '_' + str(j) + '_ok.txt')
            junkfile = os.path.join(dir, query_list[i] + '_' + str(j) + '_junk.txt')
            query = load_query(queryfile)
            good = load_list(goodfile)
            ok = load_list(okfile)
            junk = load_list(junkfile)
            good.extend(ok) # good和ok都认为是好的查询
            querys.append(query)
            q_v[query] = [tuple(good), tuple(junk)] 
    return tuple(querys), q_v

# 计算单个查询的ap值
def comput_ap(gt, rank):
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0

    intersect_size = 0
    j = 0

    for r in rank:
        # junk类型不做处理
        if r[0] in gt[1]:
            continue
        if r[0] in gt[0]:
            intersect_size += 1
        
        recall = intersect_size / float(len(gt[0]))
        precision = intersect_size / (j + 1.0)

        ap += (recall-old_recall) * ((old_precision+precision)/2.0)
        old_recall = recall
        old_precision = precision
        j += 1
    return ap

class Oxford5k(Dataset):
    def __init__(self, datadir, datapth):
        super().__init__(datadir, datapth)
        self.datadir = os.path.join(datadir, 'oxford5k')
        self.gtdir = os.path.join(datadir, 'oxford5k_txt')
        self.query_list = ['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church',
                'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']

    def get_id(self, path):
        # 获取文件基础名,不含前后缀
        pathname = os.path.splitext(os.path.basename(path))
        return pathname[0]

    def evaluate(self):
        features = torch.load(self.datapth)
        querys, q_v = get_query(self.query_list, self.gtdir)
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
def oxford5k(datadir, datapth):
    return Oxford5k(datadir, datapth)
