import os, sys

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms



class OULU_NPU_3D_HRN(Dataset):
    def __init__(self, root_dir, local_rank, part='train', transform=None):
        super(OULU_NPU_3D_HRN, self).__init__()
        # self.transform = transform
        # self.root_dir = root_dir
        # self.local_rank = local_rank
        # path_imgrec = os.path.join(root_dir, 'train.rec')
        # path_imgidx = os.path.join(root_dir, 'train.idx')
        # self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        # s = self.imgrec.read_idx(0)
        # header, _ = mx.recordio.unpack(s)
        # if header.flag > 0:
        #     self.header0 = (int(header.label[0]), int(header.label[1]))
        #     self.imgidx = np.array(range(1, int(header.label[0])))
        # else:
        #     self.imgidx = np.array(list(self.imgrec.keys))

        # BERNARDO
        # TODO: load tuples (rgb path, point cloud path, label)   # 0: spoof, 1: real
        pass


    def __getitem__(self, index):
        # idx = self.imgidx[index]
        # s = self.imgrec.read_idx(idx)
        # header, img = mx.recordio.unpack(s)
        # label = header.label
        # if not isinstance(label, numbers.Number):
        #     label = label[0]
        # label = torch.tensor(label, dtype=torch.long)
        # sample = mx.image.imdecode(img).asnumpy()
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # return sample, label

        # BERNARDO
        # TODO: return tuple (rgb, pointcloud, label)
        pass


    def __len__(self):
        # return len(self.imgidx)

        # BERNARDO
        # TODO: return length of tuples list
        pass