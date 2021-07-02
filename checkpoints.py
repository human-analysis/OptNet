# checkpoints.py

import os
import torch


class Checkpoints:
    def __init__(self, args):
        self.dir_save = args.save_dir
        self.model_filename = args.resume
        self.save_results = args.save_results

        if self.save_results and not os.path.isdir(self.dir_save):
            os.makedirs(self.dir_save)

    def latest(self, name):
        if name == 'resume':
            return self.model_filename

    def save(self, k, lam, type, model):

        torch.save(model.state_dict(),
                   '%s/iter_%d_lam_%.4f_type_%s.pth' % (self.dir_save, k,lam,type))

    def load(self, model, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            state_dict = torch.load(filename)
            model.load_state_dict(state_dict)
            return model
        raise (Exception("=> no checkpoint found at '{}'".format(filename)))
