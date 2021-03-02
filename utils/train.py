from torch.utils.tensorboard import SummaryWriter

class AverageMeter(object):
    def __init__(self, init_steps=0):
        self.steps = init_steps
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.num = 0
        self.avg = 0.0

    def step(self, val, num=1):
        self.val = val
        self.sum += num*val
        self.num += num
        self.steps += 1
        self.avg = self.sum/self.num

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log(self, loss, lr, steps):
        self.add_scalar('loss', loss.item(), steps)
        self.add_scalar('lr', lr, steps)