import os, sys
import glob
import time


def load_file_protocol(file_path):
    protocol_data = []
    with open(file_path) as f:
        line = f.readline()
        while line:
            protocol_data.append(line.strip().split(','))   # [label, video_id]
            line = f.readline()
        # print('protocol_data:', protocol_data)
        # sys.exit(0)
        
        label_real = '+1'
        label_spoof = '-1'
        for i in range(len(protocol_data)):
            protocol_data[i][0] = 1 if protocol_data[i][0] == label_real else 0
        # print('protocol_data:', protocol_data)
        # sys.exit(0)
        return protocol_data
    

def find_files(directory, extension, sort=True):
    matching_files = []

    def search_recursively(folder):
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(extension):
                    matching_files.append(os.path.join(root, file))
    search_recursively(directory)
    if sort:
        matching_files.sort()
    return matching_files


def find_neighbor_file(path_file, neighbor_ext):
    path_dir = os.path.dirname(path_file)
    neighbor_path = glob.glob(path_dir + '/*' + neighbor_ext)
    if len(neighbor_path) == 0:
        raise Exception(f'Error, no file \'*{neighbor_ext}\' found in dir \'{path_dir}\'')
    return neighbor_path[0]


def count_all_frames(protocol_data, frames_path_part, rgb_file_ext):
    num_frames = 0
    for i, (label, video_name) in enumerate(protocol_data):
        rgb_file_pattern = os.path.join(frames_path_part, video_name+'*', '*'+rgb_file_ext)
        rgb_file_paths = glob.glob(rgb_file_pattern)
        num_frames += len(rgb_file_paths)
        print(f'Counting frames - video: {i}/{len(protocol_data)-1} - num_frames: {num_frames}', end='\r')
    print('')
    return num_frames


def make_samples_list(protocol_data=[], frames_per_video=1, frames_path_part='', rgb_file_ext='', pc_file_ext='', ignore_pointcloud_files=False, level=1, attack_type_as_label=False):
    # samples_list = [None] * len(protocol_data)
    if frames_per_video > 0:
        num_frames = frames_per_video * len(protocol_data)
    else:
        num_frames = count_all_frames(protocol_data, frames_path_part, rgb_file_ext)
    samples_list = [None] * num_frames

    global_idx = 0
    for i, (label, video_name) in enumerate(protocol_data):
        # print('label:', label, '    video_name:', video_name)
        if attack_type_as_label:
            phone, session, user, file_id = video_name.split('_')
            label = int(file_id) - 1

        if level == 0:
            rgb_file_pattern = os.path.join(frames_path_part, video_name+'*'+rgb_file_ext)
        elif level == 1:
            rgb_file_pattern = os.path.join(frames_path_part, video_name+'*', '*'+rgb_file_ext)
        rgb_file_paths = glob.glob(rgb_file_pattern)
        if len(rgb_file_paths) == 0:
            raise Exception(f'Error, no file \'{rgb_file_pattern}\' found in dir \'{frames_path_part}\'')
        # rgb_file_path = rgb_file_path[0]
        for j, rgb_file_path in enumerate(rgb_file_paths):
            print(f'\'video: {i}/{len(protocol_data)-1}  -  sample: {j}/{len(rgb_file_paths)-1}  -  global_idx: {global_idx}/{num_frames-1}\'  -  ignore_pointcloud_files: {ignore_pointcloud_files}', end='\r')

            dir_sample = os.path.dirname(rgb_file_path)
            # print('rgb_file_path:', rgb_file_path)
            # print('dir_sample:', dir_sample)
            # sys.exit(0)

            if not ignore_pointcloud_files:
                pc_file_pattern = os.path.join(dir_sample, '*'+pc_file_ext)
                pc_file_path = glob.glob(pc_file_pattern)
                if len(pc_file_path) == 0:
                    raise Exception(f'Error, no file \'{pc_file_pattern}\' found in dir \'{dir_sample}\'')
                pc_file_path = pc_file_path[0]
                # print('pc_file_pattern:', pc_file_pattern)
                # print('pc_file_path:', pc_file_path)
                # sys.exit(0)
            else:
                pc_file_path = None

            one_sample = (rgb_file_path, pc_file_path, label)
            # samples_list.append(one_sample)
            samples_list[global_idx] = one_sample

            global_idx += 1
    print('')

    return samples_list


def load_file_protocol_UniAttackData(file_path):
    protocol_data = []
    with open(file_path) as f:
        line = f.readline()
        while line:     # [image label]    p1/train/000001.jpg 1
            sample = line.strip().split(' ')
            sample[1] = int(sample[1])         # convert train label to int
            protocol_data.append(sample)   
            line = f.readline()
        # print('protocol_data:', protocol_data)
        # print('len(protocol_data):', len(protocol_data))
        # sys.exit(0)
        return protocol_data


# def count_all_frames_UniAttackData(protocol_data, rgb_path='', pc_path='', rgb_file_ext='', pc_file_ext=''):
def filter_valid_samples_UniAttackData(protocol_data, rgb_path='', pc_path='', rgb_file_ext='', pc_file_ext=''):
    protocol_data_valid_samples = []
    for i, (sample, label) in enumerate(protocol_data):
        sample_name = sample.split('/')[-1].split('.')[0]

        rgb_file_pattern = os.path.join(rgb_path, sample_name+'*'+rgb_file_ext)
        rgb_file_paths = glob.glob(rgb_file_pattern)
        # print('rgb_file_paths:', rgb_file_paths)
        assert len(rgb_file_paths) <= 1, f'Error, multiple rgb files found \'{rgb_file_paths}\''

        pc_file_pattern = os.path.join(pc_path, sample_name, sample_name+'*'+pc_file_ext)
        pc_file_paths = glob.glob(pc_file_pattern)
        # print('pc_file_paths:', pc_file_paths)
        assert len(pc_file_paths) <= 1, f'Error, multiple point cloud files found \'{pc_file_paths}\''
        # sys.exit(0)

        if len(rgb_file_paths) == 1 and len(pc_file_paths) == 1:
            protocol_data_valid_samples.append([sample, label])

        print(f'Filtering valid samples - protocol_data: {i+1}/{len(protocol_data)} - protocol_data_valid_samples: {len(protocol_data_valid_samples)}', end='\r')
    print('')
    return protocol_data_valid_samples


def make_samples_list_UniAttackData(protocol_data=[], frames_per_video=1, rgb_path='', pc_path='', rgb_file_ext='', pc_file_ext='', ignore_pointcloud_files=False, level=1, attack_type_as_label=False):
    protocol_data_valid_samples = filter_valid_samples_UniAttackData(protocol_data, rgb_path, pc_path, rgb_file_ext, pc_file_ext)
    samples_list = [None] * len(protocol_data_valid_samples)

    global_idx = 0
    num_real_samples = 0
    num_spoof_samples = 0
    for i, (sample, label) in enumerate(protocol_data_valid_samples):
        print(f'Making samples list - protocol_data: {i+1}/{len(protocol_data_valid_samples)}', end='\r')
        sample_name = sample.split('/')[-1].split('.')[0]

        rgb_file_pattern = os.path.join(rgb_path, sample_name+'*'+rgb_file_ext)
        rgb_file_path = glob.glob(rgb_file_pattern)
        # print('rgb_file_path:', rgb_file_path)
        assert len(rgb_file_path) > 0, f'Error, no rgb files found with pattern \'{rgb_file_pattern}\''
        assert len(rgb_file_path) <= 1, f'Error, multiple rgb files found \'{rgb_file_path}\''
        rgb_file_path = rgb_file_path[0]

        pc_file_pattern = os.path.join(pc_path, sample_name, sample_name+'*'+pc_file_ext)
        pc_file_path = glob.glob(pc_file_pattern)
        # print('pc_file_path:', pc_file_path)
        assert len(pc_file_path) > 0, f'Error, no point cloud files found with pattern \'{pc_file_pattern}\''
        assert len(pc_file_path) <= 1, f'Error, multiple point cloud files found \'{pc_file_path}\''
        pc_file_path = pc_file_path[0]
        # sys.exit(0)

        one_sample = (rgb_file_path, pc_file_path, label)
        samples_list[global_idx] = one_sample

        if label == 0 or label == '0':
            num_real_samples += 1
        else:
            num_spoof_samples += 1

        global_idx += 1
    print('')

    return samples_list, num_real_samples, num_spoof_samples