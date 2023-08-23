import os, sys
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import utils_dataloaders as ud


class OULU_NPU_FRAMES_3D_HRN(Dataset):
    def __init__(self, root_dir, protocol_id, frames_path, img_size, part='train', local_rank=0, transform=None):
        super(OULU_NPU_FRAMES_3D_HRN, self).__init__()
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

        self.img_size = img_size
        self.protocols_path = os.path.join(root_dir, 'Protocols', 'Protocol_'+str(protocol_id))
        
        if part == 'train':
            self.root_dir_part = os.path.join(root_dir, 'train')
            self.protocol_file_path = os.path.join(self.protocols_path, 'Train.txt')
        elif part == 'val' or part == 'validation' or part == 'dev' or part == 'development':
            self.root_dir_part = os.path.join(root_dir, 'dev')
            self.protocol_file_path = os.path.join(self.protocols_path, 'Dev.txt')
        elif part == 'test':
            self.root_dir_part = os.path.join(root_dir, 'test')
            self.protocol_file_path = os.path.join(self.protocols_path, 'Test.txt')
        else:
            raise Exception(f'Error, dataset partition not recognized: \'{part}\'')

        self.frames_path_part = self.root_dir_part.replace(root_dir, frames_path)

        self.protocol_data = ud.load_file_protocol(self.protocol_file_path)

        self.rgb_file_ext = '_input_face.jpg'
        self.pc_file_ext = '_hrn_high_mesh.obj'
        self.samples_list = ud.make_samples_list(self.protocol_data, self.frames_path_part, self.rgb_file_ext, self.pc_file_ext)
        
        assert len(self.protocol_data) == len(self.samples_list), 'Error, len(self.protocol_data) must be equals to len(self.samples_list)'


    def normalize_img(self, img):
        img = np.transpose(img, (2, 0, 1))  # from (224,224,3) to (3,224,224)
        img = ((img/255.)-0.5)/0.5
        # print('img:', img)
        # sys.exit(0)
        return img

    
    def load_img(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # img_rgb = np.asarray(Image.open(img_path))
        # print('img:', img)
        # sys.exit(0)
        if (img_rgb.shape[0], img_rgb.shape[1]) != (self.img_size, self.img_size):
            img_rgb = cv2.resize(img_rgb, dsize=(self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
        return img_rgb.astype(np.float32)


    def normalize_pc(self, pc):
        pc = (pc - pc.min()) / (pc.max() - pc.min())
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


    # Based on https://github.com/youngLBW/HRN/blob/main/util/util_.py#L398
    def read_obj(self, obj_path, print_shape=False):
        with open(obj_path, 'r') as f:
            bfm_lines = f.readlines()

        vertices = []
        faces = []
        uvs = []
        vns = []
        faces_uv = []
        faces_normal = []
        max_face_length = 0
        for line in bfm_lines:
            if line[:2] == 'v ':
                vertex = [float(a) for a in line.strip().split(' ')[1:] if len(a)>0]
                vertices.append(vertex)

            if line[:2] == 'f ':
                items = line.strip().split(' ')[1:]
                face = [int(a.split('/')[0]) for a in items if len(a)>0]
                max_face_length = max(max_face_length, len(face))
                # if len(faces) > 0 and len(face) != len(faces[0]):
                #     continue
                faces.append(face)

                if '/' in items[0] and len(items[0].split('/')[1])>0:
                    face_uv = [int(a.split('/')[1]) for a in items if len(a)>0]
                    faces_uv.append(face_uv)

                if '/' in items[0] and len(items[0].split('/')) >= 3 and len(items[0].split('/')[2])>0:
                    face_normal = [int(a.split('/')[2]) for a in items if len(a)>0]
                    faces_normal.append(face_normal)

            if line[:3] == 'vt ':
                items = line.strip().split(' ')[1:]
                uv = [float(a) for a in items if len(a)>0]
                uvs.append(uv)

            if line[:3] == 'vn ':
                items = line.strip().split(' ')[1:]
                vn = [float(a) for a in items if len(a)>0]
                vns.append(vn)

        vertices = np.array(vertices).astype(np.float32)
        if max_face_length <= 3:
            faces = np.array(faces).astype(np.int32)
        else:
            print('not a triangle face mesh!')

        if vertices.shape[1] == 3:
            mesh = {
                'vertices': vertices,
                'faces': faces,
            }
        else:
            mesh = {
                'vertices': vertices[:, :3],
                'colors': vertices[:, 3:],
                'faces': faces,
            }

        if len(uvs) > 0:
            uvs = np.array(uvs).astype(np.float32)
            mesh['UVs'] = uvs

        if len(vns) > 0:
            vns = np.array(vns).astype(np.float32)
            mesh['normals'] = vns

        if len(faces_uv) > 0:
            if max_face_length <= 3:
                faces_uv = np.array(faces_uv).astype(np.int32)
            mesh['faces_uv'] = faces_uv

        if len(faces_normal) > 0:
            if max_face_length <= 3:
                faces_normal = np.array(faces_normal).astype(np.int32)
            mesh['faces_normal'] = faces_normal

        if print_shape:
            print('num of vertices', len(vertices))
            print('num of faces', len(faces))
        return mesh

    
    def flat_pc_axis_z(self, pc_data):
        for i in range(pc_data.shape[0]):
            pc_data[i, 2] = 0.  # 0=x, 1=y, 2=z
        return pc_data


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

        # Bernardo
        img_path, pc_path, label = self.samples_list[index]

        if img_path.endswith('.jpg') or img_path.endswith('.jpeg') or img_path.endswith('.png'):
            rgb_data = self.load_img(img_path)
            rgb_data = self.normalize_img(rgb_data)

        # if pc_path.endswith('.obj'):
        #     pc_data = self.read_obj(pc_path)['vertices']
        #     pc_data = self.normalize_pc(pc_data)
        # if label == 0:
        #     pc_data = self.flat_pc_axis_z(pc_data)
        pc_data = np.zeros(1)   # ONLY FOR TESTS
        
        return (rgb_data, pc_data, label)


    def __len__(self):
        # return len(self.imgidx)       # original
        return len(self.samples_list)   # Bernardo