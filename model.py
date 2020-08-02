from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils import *

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
def parse_yolo_config(cfgfile):

    # cleaning up the cfg file and creating a list of layers
    cfg_file = open(cfgfile,'r')
    lines = cfg_file.read().split('\n')
    lines = [x for x in lines if len(x) > 0 ]
    lines = [x for x in lines if x[0] !='#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}

            block['type'] = line[1:-1].rstrip()

        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    
    blocks.append(block)

    return blocks

class Emptylayer(nn.Module):
    def __init__(self):
        super(Emptylayer,self).__init__()

class Detectionlayer(nn.Module):
    def __init__(self,anchors):
        super(Detectionlayer,self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_params = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for id_,x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x["type"] == "convolutional":
            activation = x["activation"]
            try:
                batch_norm = int(x["batch_normalize"])
                bias = False
            except:
                batch_norm = 0
                bias = True
            
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias = bias)
            module.add_module("conv_{0}".format(id_),conv)

            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(id_),bn)
            
            if activation == "leaky":
                relu = nn.LeakyReLU(0.1,inplace = True)
                module.add_module("leaky_{0}",relu)
        
        elif x["type"] == "upsample":
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor= stride, mode = "nearest")
            module.add_module("upsample_{0}".format(id_),upsample)
        
        elif x["type"] == 'route':
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            
            if start > 0:
                start = start - id_
            if end > 0:
                end = end - id_
            
            route = Emptylayer()
            module.add_module('route_{0}'.format(id_),route)
            if end < 0:
                filters = output_filters[id_+ start] + output_filters[id_+ end]
            else:
                filters = output_filters[id_ + start]
        
        elif x["type"] == 'shortcut':
            shortcut = Emptylayer()
            module.add_module("shortcut_{}".format(id_),shortcut)
        
        elif x['type'] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = Detectionlayer(anchors)
            module.add_module("Detection_{}".format(id_),detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_params,module_list)

class Yolo(nn.Module):
    def __init__(self,cfgfile):
        super(Yolo,self).__init__()
        self.blocks = parse_yolo_config(cfgfile)
        self.net_params, self.module_list = create_modules(self.blocks)
    
    def forward(self,x,device):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i,module in enumerate(modules):
            module_type = (module['type'])

            if module_type == 'convolutional' or module_type == "upsample":
                x = self.module_list[i](x)
            
            elif module_type == 'route':
                layers = module["layers"]
                layers = [int(a) for a in layers]
                
                if (layers[0] > 0 ):
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1,map2),1)
            
            elif module_type == 'shortcut':
                feed = int(module['from'])
                x = outputs[i-1] + outputs[i + feed]
            
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_params['height'])
                num_classes = int(module["classes"])

                x = x.data
                x = predict_transform(x,inp_dim,anchors,num_classes,device)
                if not write:
                    detections = x
                    write = 1
                
                else:
                    detections = torch.cat((detections,x),1)
            
            outputs[i] = x
        
        return detections
    
    def load_weights(self,weightfile):

        fp = open(weightfile,"rb")

        header = np.fromfile(fp,dtype = np.int32,count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp,dtype= np.float32)

        ptr = 0

        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            if module_type == 'convolutional':
                model = self.module_list[i]
                try: batch_norm = int(self.blocks[i+1]["batch_normalize"])
                except: batch_norm = 0
            
                conv = model[0]

                if batch_norm:
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:

                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)
                
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    