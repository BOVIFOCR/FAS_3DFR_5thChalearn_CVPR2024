import sys, os
import argparse
import numpy as np

from matplotlib import pyplot as plt


def plot_protocol_data(img_paths, labels, title, path_save):
    x = np.arange(len(labels))
    
    # fig, axes = plt.subplots(1, 1)
    width, height = 6, 3
    fig, axes = plt.subplots(1, 1, figsize=(width, height))  # Adjust size here
    fig.suptitle(title)

    interval = int(len(labels)/10)
    img_paths_xticks = [path if i%interval==0 or i+1==len(labels) else '' for i, path in enumerate(img_paths)]
    
    axes.plot(x, labels, '.-')
    axes.set_xlabel('Samples')
    axes.xaxis.set_ticks(x)
    axes.set_xticklabels(img_paths_xticks, rotation=90)

    axes.set_ylabel('Labels\n(0=real; 1=spoof)')
    axes.set_ylim([-0.25, 1.25])
    axes.set_yticks([0, 1])

    fig.tight_layout()
    fig.savefig(path_save)   # save the figure to file
    plt.close(fig)


def load_data(file_path):
    all_data = []
    paths = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()  # Split the line by whitespace
            img_path = line[0]  # First element is image path
            label = int(line[1])  # Second element is label, converted to int
            all_data.append([img_path, label])  # Append [img_path, label] to all_data list
            paths.append(img_path)
            labels.append(label)
    return all_data, paths, labels


def main(args):
    print(f'Loading data from file \'{args.input}\'')
    all_data, paths, labels = load_data(args.input)
    # print('labels:', labels)
    
    paths_np = np.array(paths)
    labels_np = np.array(labels)
    # print('paths_np:', paths_np)
    # print('type(paths_np):', type(paths_np))
    # print('labels_np:', labels_np)

    title = 'Dataset UniAttackData - part: ' + args.input.split('/')[-2]
    file_name = args.type + '_' + '_'.join(args.input.split('/')[-3:])  + '.png'
    path_save = os.path.join(os.path.dirname(args.input), file_name) 
    print(f'Saving chart \'{path_save}\'')
    plot_protocol_data(paths, labels_np, title, path_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset analyser")
    parser.add_argument("--type", type=str, default='plot-true-labels', help='Options: plot-true-labels, plot-pred-labels, ')
    parser.add_argument("--input", type=str, default='/datasets1/bjgbiesseck/liveness/fas_cvpr2024/UniAttackData/phase1/p1/train_label.txt')
    
    main(parser.parse_args())