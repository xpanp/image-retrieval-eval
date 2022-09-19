from .registry import register_dataset
from .dataset import Dataset
from pathlib import Path
import torch
import os

# 获取列表的第二个元素
def take_second(elem):
    return elem[1]

def get_groundtruth(gtfile):
    """ Read datafile holidays_images.dat and output a dictionary
    mapping queries to the set of positive results (plus a list of all
    images)"""
    gt={}
    allnames=set()
    for line in open(gtfile,"r"):
        imname=line.strip()
        allnames.add(imname)
        imno=int(imname[:-len(".jpg")])    
        if imno%100==0:
            gt_results=set()
            gt[imname]=gt_results
        else:
            gt_results.add(imname)
        
    return allnames,gt
    
def score_ap_from_ranks_1(ranks, nres):
    """ Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset  
    """
    
    # accumulate trapezoids in PR-plot
    ap=0.0

    # All have an x-size of:
    recall_step=1.0/nres
        
    for ntp,rank in enumerate(ranks):
        
        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank==0: precision_0=1.0
        else:       precision_0=ntp/float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1=(ntp+1)/float(rank+1)
        
        ap+=(precision_1+precision_0)*recall_step/2.0
            
    return ap

class Holidays(Dataset):
    def __init__(self, datadir, datapth):
        super().__init__(datadir, datapth)
        self.datadir = os.path.join(datadir, 'jpg')
        self.gtfile = os.path.join(datadir, 'eval_holidays/holidays_images.dat')

    def get_id(self, path):
        name = Path(path).name
        return name
        # res = os.path.split(path)
        # return res[-1]

    def evaluate(self):
        features = torch.load(self.datapth)
        _, gt = get_groundtruth(self.gtfile)
        sum_ap = 0.
        n = 0
        for i in range(len(features)):
            # print(features[i][1].shape)
            imname = features[i][0]
            imno = int(imname[:-len(".jpg")])
            # 可以整除100的，为查询图像
            if imno % 100 != 0:
                print(i, "/", len(features), ", jpg:", features[i][0], "-------skip")
                continue
            
            print(i, "/", len(features), ", jpg:", features[i][0])

            # 获取所有相似度
            results = []
            for j in range(len(features)):
                score = torch.cosine_similarity(features[i][1], features[j][1], dim=1)
                results.append((features[j][0], score))
            results.sort(key=take_second, reverse=True)

            gt_results=gt.pop(features[i][0])
            tp_ranks=[]
            print_res=[]
            for j in range(len(results)):
                if results[j][0] in gt_results:
                    tp_ranks.append(j - 1)
                    print_res.append((j, results[j][0], results[j][1]))
            print(i, "/", len(features), ", jpg:", features[i][0], ", res:", print_res)

            sum_ap += score_ap_from_ranks_1(tp_ranks, len(gt_results))
            n+=1
        
        if gt:
            print("---------no result for queries", gt.keys())

        print("mAP: %.5f" % (sum_ap/n))

@register_dataset
def holidays(datadir, datapth):
    return Holidays(datadir, datapth)
