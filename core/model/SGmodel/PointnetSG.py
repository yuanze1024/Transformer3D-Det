import torch.nn as nn
from core.model.task_basemodel.backbone.base_model import base_module
from core.model.Pointnet import model_entry
from .utils import split_obj, split_rel


class PointNetSG(base_module):
    def __init__(self, config):
        super(PointNetSG, self).__init__()
        self.params = []
        self.obj_module = model_entry(config.objmodel)
        self.rel_module = model_entry(config.relmodel)
        assert self.obj_module is not None, 'obj module is None'
        assert self.rel_module is not None, 'rel module is None'
        # print(self.obj_module.parameters())
        self.init_relu = 'relu'
        self.init_params(nn.BatchNorm2d, init_type='kaiming_normal')

    def _forward(self, input):
        # print(xyz.shape)
        output = {}
        if 'object_point_set' in input.keys():
            output['obj_result'] = self.obj_module({'point_set': input['obj_point_set']})
        if 'rel_point_set' in input.keys():
            output['rel_result'] = self.rel_module({'point_set': input['rel_point_set']})
        return output

    def _before_forward(self, input):
        if 'object_point_set' in input.keys():
            input = split_obj(input)
        if 'rel_point_set' in input.keys():
            input = split_rel(input)
        return input

    def calculate_loss(input, output):
        loss = 0
        if 'rel_point_set' in input.keys():
            rel_loss = 0
        if 'object_point_set' in input.keys():
            input = split_obj(input)
        loss += rel_loss
        # input['one_hot_rel_target'] = rel_mask
        # input['rel_mask'] = rel_mask
        # input['rel_points'] = rel_points  # splited
        # input['rel_idx'] = rel_idx
        # input['object_target'] = object_target
        # input['object_points'] = object_points  # splited
        # input['object_idx'] = object_idx
        pass


if __name__ == "__main__":
    import sys
    import os
    from easydict import EasyDict

    config = {
        'num_output': 83,
        'normal_channel': False
    }
    config = EasyDict(config)
    print(os.getcwd())
    net = PointNetSG(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
