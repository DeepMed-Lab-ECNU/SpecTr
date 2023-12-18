import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def B_postprocess_output(network_output: torch.Tensor):
    out = network_output.cpu().data.numpy()
    #pred = np.argmax(out, axis=1).tolist()
    pred = np.where(out>0.5,1,0)
    return pred


if __name__ == '__main__':
    out = torch.tensor([[1,2,3],[8,1,4]])
#    print(postprocess_output(out))