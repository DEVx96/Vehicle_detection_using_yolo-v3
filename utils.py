from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def predict_transform(prediction, inp_dim, anchors, num_classes, device):

    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    
    x_offset = x_offset.to(device)
    y_offset = y_offset.to(device)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)

    anchors = anchors.to(device)

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction

def bbox_iou(box1, box2):
   
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou



def write_results(preds, obs_thresh, num_classes, nms_thresh = 0.4):

    obs_mask = (preds[:,:,4] > obs_thresh).float().unsqueeze(2)
    preds = preds * obs_mask

    box_corner = preds.new(preds.shape)
    box_corner[:,:,0] = (preds[:,:,0] - preds[:,:,2]/2)
    box_corner[:,:,1] = (preds[:,:,1] - preds[:,:,3]/2)
    box_corner[:,:,2] = (preds[:,:,0] + preds[:,:,2]/2) 
    box_corner[:,:,3] = (preds[:,:,1] + preds[:,:,3]/2)
    preds[:,:,:4] = box_corner[:,:,:4]    

    bs = preds.size(0)

    write = False

    for i in range(bs):
        img_pred = preds[i]
        max_conf,max_conf_score = torch.max(img_pred[:,5:5 + num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (img_pred[:,:5],max_conf,max_conf_score)
        img_pred = torch.cat(seq,1)

        non_zer_ind = (torch.nonzero(img_pred[:,4]))
        try:
            img_pred_ = img_pred[non_zer_ind.squeeze(),:].view(-1,7)
        except:
            continue
        if img_pred_.shape[0] == 0:
            continue

        img_classes = unique(img_pred_[:,-1])

        for cls_ in img_classes:
            cls_mask = img_pred_*(img_pred_[:,-1] == cls_).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            img_pred_class = img_pred_[class_mask_ind].view(-1,7)
            
            
            conf_sort_index = torch.sort(img_pred_class[:,4], descending = True )[1]
            img_pred_class = img_pred_class[conf_sort_index]
            idx = img_pred_class.size(0)   
            
            for i in range(idx):
               
                try:
                    ious = bbox_iou(img_pred_class[i].unsqueeze(0), img_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                iou_mask = (ious < nms_thresh).float().unsqueeze(1)
                img_pred_class[i+1:] *= iou_mask       
            
                non_zero_ind = torch.nonzero(img_pred_class[:,4]).squeeze()
                img_pred_class = img_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = img_pred_class.new(img_pred_class.size(0), 1).fill_(i)      
            seq = batch_ind, img_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0
    
def letterbox_image(img, inp_dim):

    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

