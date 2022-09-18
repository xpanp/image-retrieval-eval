from abc import ABC, abstractmethod
import os

# 输入文件夹路径，返回文件列表，不可嵌套获取
def dir_2_filelist(d):
    names = os.listdir(d)
    paths = []  # 所有图片路径
    for name in names:
        paths.append(os.path.join(d, name))
    return paths

class Dataset(ABC):
    def __init__(self, datadir, datapth):
        self.datapth = datapth
        self.datadir = datadir

    def get_filelist(self):
        return dir_2_filelist(self.datadir)

    def get_pth_path(self):
        return self.datapth

    @abstractmethod
    def get_id(self, path):
        pass

    @abstractmethod
    def evaluate(self):
        pass  
