from .registry import register_dataset
from .dataset import Dataset
from .oxford5k import load_list, comput_ap
import torch
import os

# 输入文件夹路径，返回文件列表
# paris文件夹多一层文件夹
def dir_2_filelist(dir, destroy):
    names = os.listdir(dir)
    paths = []  # 所有图片路径
    for name in names:
        files = os.listdir(os.path.join(dir, name))
        for file in files:
            if file in destroy:
                continue
            paths.append(os.path.join(dir, name, file))
    return paths

def load_query(filename):
    # 如: paris_sacrecoeur_000437 367.000000 137.000000 727.000000 511.000000
    # 获取: paris_sacrecoeur_000437
    with open(filename, 'r') as f:
        s = f.readlines()
    return s[0].split(' ')[0]

def get_query(query_list, dir):
    querys = []
    q_v = {}
    for i in range(len(query_list)):
        for j in range(1, 6):
            queryfile = os.path.join(dir, query_list[i] + '_' + str(j) + '_query.txt')
            # goodfile = os.path.join(dir, query_list[i] + '_' + str(j) + '_good.txt') # 该文件夹都是空的
            okfile = os.path.join(dir, query_list[i] + '_' + str(j) + '_ok.txt')
            junkfile = os.path.join(dir, query_list[i] + '_' + str(j) + '_junk.txt')
            query = load_query(queryfile)
            ok = load_list(okfile)
            junk = load_list(junkfile)
            querys.append(query)
            q_v[query] = [tuple(ok), tuple(junk)] 
    return tuple(querys), q_v

class Paris6k(Dataset):
    def __init__(self, datadir, datapth):
        super().__init__(datadir, datapth)
        self.datadir = os.path.join(datadir, 'paris')
        self.gtdir = os.path.join(datadir, 'txt')
        self.query_list = ['defense', 'eiffel', 'invalides', 'louvre', 'moulinrouge',
                'museedorsay', 'notredame', 'pantheon', 'pompidou', 'sacrecoeur', 'triomphe']
        # 损坏图片
        self.destroy = ('paris_louvre_000136.jpg', 'paris_louvre_000146.jpg', 'paris_moulinrouge_000422.jpg', 'paris_museedorsay_001059.jpg',
            'paris_notredame_000188.jpg', 'paris_pantheon_000284.jpg', 'paris_pantheon_000960.jpg', 'paris_pantheon_000974.jpg',
            'paris_pompidou_000195.jpg', 'paris_pompidou_000196.jpg', 'paris_pompidou_000201.jpg', 'paris_pompidou_000467.jpg',
            'paris_pompidou_000640.jpg', 'paris_sacrecoeur_000299.jpg', 'paris_sacrecoeur_000330.jpg', 'paris_sacrecoeur_000353.jpg',
            'paris_triomphe_000662.jpg', 'paris_triomphe_000833.jpg', 'paris_triomphe_000863.jpg', 'paris_triomphe_000867.jpg')

    def get_id(self, path):
        # 获取文件基础名,不含前后缀
        pathname = os.path.splitext(os.path.basename(path))
        return pathname[0]

    def get_filelist(self):
        return dir_2_filelist(self.datadir, self.destroy)

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
def paris6k(datadir, datapth):
    return Paris6k(datadir, datapth)
