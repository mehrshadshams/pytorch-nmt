SOS_WORD = '<SOS>'
EOS_WORD = '<EOS>'
PAD_WORD = '<PAD>'


def tensor2np(tensor):
    return tensor.data.cpu().numpy()


class AverageMeter(object):
    """
        Computes and stores the average and current value
        Borrowed from ImageNet training in PyTorch project
        https://github.com/pytorch/examples/tree/master/imagenet
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

