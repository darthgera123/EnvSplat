import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.spec_utils import SpecularNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.nerf_utils import NeRFNetwork

class SpecularModel:
    def __init__(self,model='specular'):
        if model == 'specular':
            self.specular = SpecularNetwork().cuda()
        elif model == 'hash':  
            self.specular = NeRFNetwork(
            encoding="hashgrid",
            num_layers=2,
            num_layers_color=3,
            bound=2,
            cuda_ray=False,
            density_scale=1,
            min_near=0.2,
            density_thresh=10,
            bg_radius=-1,
            ).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    # def step(self, xyz, viewdir):
    #     return self.specular(xyz, viewdir)
    def step(self, xyz, viewdir,lightdir=None):
        if lightdir == None:
            return self.specular(xyz,viewdir)
        else:
            return self.specular(xyz, viewdir,lightdir)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.specular.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "specular"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.specular_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                      lr_final=training_args.position_lr_final,
                                                      lr_delay_mult=training_args.position_lr_delay_mult,
                                                      max_steps=training_args.specular_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "specular/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.specular.state_dict(), os.path.join(out_weights_path, 'specular.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "specular"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "specular/iteration_{}/specular.pth".format(loaded_iter))
        self.specular.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "specular":
                lr = self.specular_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
