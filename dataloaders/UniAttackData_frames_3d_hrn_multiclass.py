import os, sys
import glob
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

from . import utils_dataloaders as ud


class UniAttackData_FRAMES_3D_HRN_MULTICLASS(Dataset):
    def __init__(self, root_dir, protocol_id, rgb_path, pc_path, img_size, frames_per_video=1, part='train', role='train', percent=0.6, ignore_pointcloud_files=False, protocol_data=None, filter_valid_samples=True, local_rank=0, transform=None):
        super(UniAttackData_FRAMES_3D_HRN_MULTICLASS, self).__init__()
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

        self.rgb_path = rgb_path
        self.pc_path = pc_path
        self.rgb_file_ext = '.png'
        self.pc_file_ext = '_hrn_high_mesh_10000points.npy'  # binary file

        self.img_size = img_size
        self.frames_per_video = frames_per_video

        if type(protocol_id) is list:
            self.protocols_path = [os.path.join(root_dir, prot_id) for prot_id in protocol_id]

            if part == 'train':
                # self.protocol_file_path = [os.path.join(prot_path, 'train_label.txt') for prot_path in self.protocols_path]
                self.protocol_file_path = [os.path.join(prot_path, 'train_multi_label.txt') for prot_path in self.protocols_path]
            elif part == 'val' or part == 'validation' or part == 'dev' or part == 'development':
                # self.protocol_file_path = [os.path.join(prot_path, 'dev.txt') for prot_path in self.protocols_path]
                # self.protocol_file_path = [os.path.join(prot_path, 'dev_label.txt') for prot_path in self.protocols_path]
                self.protocol_file_path = [os.path.join(prot_path, 'dev_multi_label.txt') for prot_path in self.protocols_path]
            elif part == 'test':
                # self.protocol_file_path = [os.path.join(prot_path, 'dev.txt') for prot_path in self.protocols_path]
                self.protocol_file_path = [os.path.join(prot_path, 'test.txt') for prot_path in self.protocols_path]
            else:
                raise Exception(f'Error, dataset partition not recognized: \'{part}\'')

        elif os.path.isfile(protocol_id):
            self.protocol_file_path = [protocol_id]
        

        for i in range(len(self.protocol_file_path)):
            assert os.path.isfile(self.protocol_file_path[i]), f'Error, protocol file not found \'{self.protocol_file_path[i]}\''
        assert os.path.isdir(self.rgb_path), f'Error, rgb path not found \'{self.rgb_path}\''
        assert os.path.isdir(self.pc_path), f'Error, point clouds path not found \'{self.pc_path}\''

        self.protocol_data = protocol_data
        if self.protocol_data is None:
            self.protocol_data = [ud.load_file_protocol_UniAttackData(prot_path) for prot_path in self.protocol_file_path]
            self.protocol_data = [sample for prot_data in self.protocol_data for sample in prot_data]  # merge all samples into one list

            if filter_valid_samples:
                self.protocol_data = ud.filter_valid_samples_UniAttackData(self.protocol_data, self.rgb_path, self.pc_path, self.rgb_file_ext, self.pc_file_ext)
        # print('self.protocol_data:', self.protocol_data)
        # print('len(self.protocol_data):', len(self.protocol_data))
        # sys.exit(0)

        if percent < 1:
            if role == 'train':
                idx_start = 0
                idx_end = int(np.ceil(len(self.protocol_data) * percent))
            elif role == 'val' or role == 'validation' or role == 'dev' or role == 'development':
                idx_start = int(np.ceil(len(self.protocol_data) * (1-percent)))
                idx_end = len(self.protocol_data)
            # elif role == 'test':
            #     idx_start = int(len(self.protocol_data) * 0.8)
            #     idx_end = len(self.protocol_data)
            random.shuffle(self.protocol_data)
            print(f'{role} percent: {percent} of {len(self.protocol_data)}    idx_start: {idx_start}    idx_end: {idx_end}')
            self.protocol_data = self.protocol_data[idx_start:idx_end]
            print('    new len(self.protocol_data):', len(self.protocol_data))

        self.samples_list, self.num_real_samples, self.num_spoof_samples = ud.make_samples_list_UniAttackData(self.protocol_data, frames_per_video, self.rgb_path, self.pc_path, self.rgb_file_ext, self.pc_file_ext, ignore_pointcloud_files)
        self.indices = np.random.choice(10000, 2500, replace=False)

        print(f'    num_real_samples: {self.num_real_samples}    num_spoof_samples: {self.num_spoof_samples}')
        # assert len(self.protocol_data) == len(self.samples_list), 'Error, len(self.protocol_data) must be equals to len(self.samples_list)'


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
        pc[:,2] = pc[:,2] + centroid[2]
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


    # Based on https://github.com/youngLBW/HRN/blob/main/util/util_.py#L343
    def write_obj(self, save_path, vertices, faces=None, UVs=None, faces_uv=None, normals=None, faces_normal=None, texture_map=None, save_mtl=False, vertices_color=None):
        save_dir = os.path.dirname(save_path)
        save_name = os.path.splitext(os.path.basename(save_path))[0]

        if save_mtl or texture_map is not None:
            if texture_map is not None:
                cv2.imwrite(os.path.join(save_dir, save_name + '.jpg'), texture_map)
            with open(os.path.join(save_dir, save_name + '.mtl'), 'w') as wf:
                wf.write('# Created by HRN\n')
                wf.write('newmtl material_0\n')
                wf.write('Ka 1.000000 0.000000 0.000000\n')
                wf.write('Kd 1.000000 1.000000 1.000000\n')
                wf.write('Ks 0.000000 0.000000 0.000000\n')
                wf.write('Tr 0.000000\n')
                wf.write('illum 0\n')
                wf.write('Ns 0.000000\n')
                wf.write('map_Kd {}\n'.format(save_name + '.jpg'))

        with open(save_path, 'w') as wf:
            if save_mtl or texture_map is not None:
                wf.write("# Create by HRN\n")
                wf.write("mtllib ./{}.mtl\n".format(save_name))

            if vertices_color is not None:
                for i, v in enumerate(vertices):
                    wf.write('v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], vertices_color[i][0], vertices_color[i][1], vertices_color[i][2]))
            else:
                for v in vertices:
                    wf.write('v {} {} {}\n'.format(v[0], v[1], v[2]))

            if UVs is not None:
                for uv in UVs:
                    wf.write('vt {} {}\n'.format(uv[0], uv[1]))

            if normals is not None:
                for vn in normals:
                    wf.write('vn {} {} {}\n'.format(vn[0], vn[1], vn[2]))

            if faces is not None:
                for ind, face in enumerate(faces):
                    if faces_uv is not None or faces_normal is not None:
                        if faces_uv is not None:
                            face_uv = faces_uv[ind]
                        else:
                            face_uv = face
                        if faces_normal is not None:
                            face_normal = faces_normal[ind]
                        else:
                            face_normal = face
                        row = 'f ' + ' '.join(['{}/{}/{}'.format(face[i], face_uv[i], face_normal[i]) for i in range(len(face))]) + '\n'
                    else:
                        row = 'f ' + ' '.join(['{}'.format(face[i]) for i in range(len(face))]) + '\n'
                    wf.write(row)

    
    def flat_pc_axis_z(self, pc_data):
        for i in range(pc_data.shape[0]):
            pc_data[i, 2] = 0.  # 0=x, 1=y, 2=z
        return pc_data


    def sample_points(self, arr, n=2500):
        if n > len(arr):
            return arr
        # indices = np.random.choice(len(arr), n, replace=False)
        indices = self.indices
        selected_elements = arr[indices]
        return selected_elements


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
        if len(self.samples_list[index]) == 4:
            img_path, pc_path, label, label2 = self.samples_list[index]
        elif len(self.samples_list[index]) == 3:
            img_path, pc_path, label = self.samples_list[index]
        elif len(self.samples_list[index]) == 2:
            img_path, label = self.samples_list[index]
            pc_path = None

        if img_path.endswith('.jpg') or img_path.endswith('.jpeg') or img_path.endswith('.png'):
            rgb_data = self.load_img(img_path)
            rgb_data = self.normalize_img(rgb_data)

        if not pc_path is None:
            if pc_path.endswith('.obj'):
                pc_data = self.read_obj(pc_path)['vertices']
            elif pc_path.endswith('.npy'):
                pc_data = np.load(pc_path)
                # pc_data = np.load(pc_path).astype(np.float32)
            pc_data = self.normalize_pc(pc_data)
            pc_data = self.sample_points(pc_data, n=2500)

            if label == 1:   # in dataset UniAttackData 0=real, 1=spoof
                pc_data = self.flat_pc_axis_z(pc_data)
        else:
            # pc_data = torch.tensor(0)
            pc_data = torch.zeros((2500,3))

        # save_path = f'./pointcloud_index={index}_label={label}.obj'
        # self.write_obj(save_path, pc_data)
        
        return (rgb_data, pc_data, label, label2)
        # return (img_path, rgb_data, pc_path, pc_data, label)


    def __len__(self):
        # return len(self.imgidx)       # original
        return len(self.samples_list)   # Bernardo