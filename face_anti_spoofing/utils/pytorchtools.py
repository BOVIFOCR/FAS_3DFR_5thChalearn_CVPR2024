import numpy as np
import math
import torch

# source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0.05, max_epochs=100, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.max_epochs = max_epochs
        self.path = path
        self.trace_func = trace_func

        self.scores = np.zeros((self.max_epochs), dtype=float)
        self.scores_smooth = np.zeros((self.max_epochs), dtype=float)
        self.mov_avg_idx = 0
        self.mov_avg = 0


    def smooth_array(self, scalars, weight=0.99):
        """
        EMA implementation according to
        https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
        """
        last = 0
        smoothed = np.zeros((len(scalars),), dtype=float)
        num_acc = 0
        for i, next_val in enumerate(scalars):
            last = last * weight + (1 - weight) * next_val
            num_acc += 1
            # de-bias
            debias_weight = 1
            if weight != 1:
                debias_weight = 1 - math.pow(weight, num_acc)
            smoothed_val = last / debias_weight
            smoothed[i] = smoothed_val

        return smoothed


    def __call__(self, val_loss, smooth=True):
        if smooth:
            self.scores[self.mov_avg_idx] = val_loss
            self.mov_avg_idx += 1

            self.scores_smooth[:self.mov_avg_idx] = self.smooth_array(self.scores[:self.mov_avg_idx], weight=0.99)
            val_loss = self.scores_smooth[self.mov_avg_idx-1]

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score > self.best_score - (self.best_score * self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.trace_func(f'EarlyStopping new best found: {score} (past: {self.best_score})')
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

        return score



    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
