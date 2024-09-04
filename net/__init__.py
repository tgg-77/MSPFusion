import torch
import os
from .densefuse import DenseFuse_net
from .HSCNN_Plus import HSCNN_Plus
from .fusion_layer import Fusion_network
from .IFCNN import myIFCNN
from .MSPFusion import MSPFusion


def model_generator(method, pretrained_model_path=None):
    if method == 'densefuse':
        model = DenseFuse_net(input_nc=1, output_nc=1).cuda()
    elif method == 'hscnn':
        model = HSCNN_Plus().cuda()
    elif method == 'fusion_layer':
        model = Fusion_network([64], 'res').cuda()
    elif method == 'ifcnn':
        model = myIFCNN(fuse_scheme=0)
    elif method == 'MSPFusion':
        model = MSPFusion()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
