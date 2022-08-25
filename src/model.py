import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import timm

## DeiT-Tiny ##
class FeaturesDeitTiny_distilled(nn.Module):
    """
    backbone: Deit-Tiny
    """

    def __init__(self):
        super(FeaturesDeitTiny_distilled, self).__init__()
        deittiny = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        self.features = deittiny
        #  modules = list(deittiny.children())[:-1]
        #  self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features.forward_features(x)
        output = torch.cat(output,dim=-1)
        output = output.view(output.size()[0], -1)
        return output

## DeiT-Small ##
class FeaturesDeitSmall_distilled(nn.Module):
    """
    backbone: DeiT-Small
    """

    def __init__(self):
        super(FeaturesDeitSmall_distilled, self).__init__()
        deitsmall = torch.hub.load('facebookresearch/deit:main', 'deit_small_distilled_patch16_224', pretrained=True)
        self.features = deitsmall
        #  modules = list(deittiny.children())[:-1]
        #  self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features.forward_features(x)
        output = torch.cat(output,dim=-1)
        output = output.view(output.size()[0], -1)
        return output

## DeiT-Base ##
class FeaturesDeitBase_distilled(nn.Module):
    """
    backbone: DeiT-Base
    """

    def __init__(self):
        super(FeaturesDeitBase_distilled, self).__init__()
        deitbase = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)
        self.features = deitbase
        #  modules = list(deittiny.children())[:-1]
        #  self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features.forward_features(x)
        output = torch.cat(output,dim=-1)
        output = output.view(output.size()[0], -1)
        return output

## ViT-Tiny ##
class FeaturesDeitTiny(nn.Module):
    """
    backbone: ViT-Tiny
    """

    def __init__(self):
        super(FeaturesDeitTiny, self).__init__()
        deittiny = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.features = deittiny
        #  modules = list(deittiny.children())[:-1]
        #  self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features.forward_features(x)
        output = output.view(output.size()[0], -1)
        return output

## ViT-Small ##
class FeaturesDeitSmall(nn.Module):
    """
    backbone: ViT-Small
    """

    def __init__(self):
        super(FeaturesDeitSmall, self).__init__()
        deitsmall = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
        self.features = deitsmall

    def forward(self, x):
        output = self.features.forward_features(x)
        output = output.view(output.size()[0], -1)
        return output

## ViT-Base
class FeaturesDeitBase(nn.Module):
    """
    backbone: ViT-Base
    """

    def __init__(self):
        super(FeaturesDeitBase, self).__init__()
        deitbase = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        self.features = deitbase

    def forward(self, x):
        output = self.features.forward_features(x)
        output = output.view(output.size()[0], -1)
        return output


## ViT-Tiny +SIE from TransReID (He et al. 2021) ##
class FeaturesDeitTiny_sie(nn.Module):
    """
    backbone: ViT-Tiny
    """

    def __init__(self):
        super(FeaturesDeitTiny_sie, self).__init__()
        deittiny = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.features = deittiny
        self.num_cam = 9
        self.sie_xishu = 3.0
        self.sie_embed = nn.Parameter(torch.zeros(self.num_cam, 1, deittiny.embed_dim))
        timm.models.layers.trunc_normal_(self.sie_embed,std=.02)
    
    def forward_features(self,x,cam):
        cam_int = (cam - 1).squeeze()
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.features.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.features.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed + self.sie_xishu * self.sie_embed[cam_int])
        x = self.features.blocks(x)
        x = self.features.norm(x)
        return x[:,0]

    def forward(self, x, cam):
        output = self.forward_features(x, cam)
        output = output.view(output.size()[0], -1)
        return output

## ViT-Tiny + P3DE ##
class FeaturesDeitTiny_P3DE(nn.Module):
    """
    backbone: ViT-Tiny
    """

    def __init__(self,grid):
        super(FeaturesDeitTiny_P3DE, self).__init__()
        deittiny = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.grid = grid
        self.features = deittiny
        self.num_cam = 9
        self.x_max = int(1920 / self.grid)
        self.y_max = int(1080 / self.grid)
        self.x_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.x_max, 1, deittiny.embed_dim))
        self.y_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.y_max, 1, deittiny.embed_dim))
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, deittiny.embed_dim))
        timm.models.layers.trunc_normal_(self.x_cam_token)
        timm.models.layers.trunc_normal_(self.y_cam_token)
        timm.models.layers.trunc_normal_(self.z_cam_token)
    
    def forward_features(self,x,cam,pos):
        cam_int = (cam - 1).squeeze()
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.features.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.features.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        pos_x = ((pos[:,0] + pos[:,2]) / (2.*self.grid)).type(torch.int64)
        pos_y = ((pos[:,1] + pos[:,3]) / (2.*self.grid)).type(torch.int64)
        x_cam_token = torch.index_select(self.x_cam_token,0,cam_int*self.x_max + pos_x)
        y_cam_token = torch.index_select(self.y_cam_token,0,cam_int*self.y_max + pos_y)
        z_cam_token = torch.index_select(self.z_cam_token,0,cam_int)
        x = torch.cat((x,x_cam_token,y_cam_token,z_cam_token),dim=1)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x = torch.cat((x[:,0],x[:,-3],x[:,-2],x[:,-1]),dim=-1)
        return x

    def forward(self, x, cam, pos):
        output = self.forward_features(x, cam, pos)
        output = output.view(output.size()[0], -1)
        return output

## ViT-Small + P3DE ##
class FeaturesDeitSmall_P3DE(nn.Module):
    """
    backbone: ViT-Small
    """

    def __init__(self,grid):
        super(FeaturesDeitSmall_P3DE, self).__init__()
        deitsmall = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
        self.grid = grid
        self.features = deitsmall
        self.num_cam = 9
        self.x_max = int(1920 / self.grid)
        self.y_max = int(1080 / self.grid)
        self.x_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.x_max, 1, deitsmall.embed_dim))
        self.y_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.y_max, 1, deitsmall.embed_dim))
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, deitsmall.embed_dim))
        timm.models.layers.trunc_normal_(self.x_cam_token)
        timm.models.layers.trunc_normal_(self.y_cam_token)
        timm.models.layers.trunc_normal_(self.z_cam_token)
    
    def forward_features(self,x,cam,pos):
        cam_int = (cam - 1).squeeze()
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.features.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.features.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        pos_x = ((pos[:,0] + pos[:,2]) / (2.*self.grid)).type(torch.int64)
        pos_y = ((pos[:,1] + pos[:,3]) / (2.*self.grid)).type(torch.int64)
        x_cam_token = torch.index_select(self.x_cam_token,0,cam_int*self.x_max + pos_x)
        y_cam_token = torch.index_select(self.y_cam_token,0,cam_int*self.y_max + pos_y)
        z_cam_token = torch.index_select(self.z_cam_token,0,cam_int)
        x = torch.cat((x,x_cam_token,y_cam_token,z_cam_token),dim=1)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x = torch.cat((x[:,0],x[:,-3],x[:,-2],x[:,-1]),dim=-1)
        return x

    def forward(self, x, cam, pos):
        output = self.forward_features(x, cam, pos)
        output = output.view(output.size()[0], -1)
        return output

## ViT-Base + P3DE ##
class FeaturesDeitBase_P3DE(nn.Module):
    """
    backbone: ViT-Base
    """

    def __init__(self,grid):
        super(FeaturesDeitBase_P3DE, self).__init__()
        deitbase = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        self.grid = grid
        self.features = deitbase
        self.num_cam = 9
        self.x_max = int(1920 / self.grid)
        self.y_max = int(1080 / self.grid)
        self.x_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.x_max, 1, deitbase.embed_dim))
        self.y_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.y_max, 1, deitbase.embed_dim))
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, deitbase.embed_dim))
        timm.models.layers.trunc_normal_(self.x_cam_token)
        timm.models.layers.trunc_normal_(self.y_cam_token)
        timm.models.layers.trunc_normal_(self.z_cam_token)
    
    def forward_features(self,x,cam,pos):
        cam_int = (cam - 1).squeeze()
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.features.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.features.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        pos_x = ((pos[:,0] + pos[:,2]) / (2.*self.grid)).type(torch.int64)
        pos_y = ((pos[:,1] + pos[:,3]) / (2.*self.grid)).type(torch.int64)
        x_cam_token = torch.index_select(self.x_cam_token,0,cam_int*self.x_max + pos_x)
        y_cam_token = torch.index_select(self.y_cam_token,0,cam_int*self.y_max + pos_y)
        z_cam_token = torch.index_select(self.z_cam_token,0,cam_int)
        x = torch.cat((x,x_cam_token,y_cam_token,z_cam_token),dim=1)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x = torch.cat((x[:,0],x[:,-3],x[:,-2],x[:,-1]),dim=-1)
        return x

    def forward(self, x, cam, pos):
        output = self.forward_features(x, cam, pos)
        output = output.view(output.size()[0], -1)
        return output

## DeiT-Tiny + P3DE ##
class FeaturesDeitTiny_distilled_P3DE(nn.Module):
    """
    backbone: DeiT-Tiny
    """

    def __init__(self,grid):
        super(FeaturesDeitTiny_distilled_P3DE, self).__init__()
        deittiny = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        self.grid = grid
        self.features = deittiny
        self.num_cam = 9
        self.x_max = int(1920 / self.grid)
        self.y_max = int(1080 / self.grid)
        self.x_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.x_max, 1, deittiny.embed_dim))
        self.y_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.y_max, 1, deittiny.embed_dim))
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, deittiny.embed_dim))
        timm.models.layers.trunc_normal_(self.x_cam_token)
        timm.models.layers.trunc_normal_(self.y_cam_token)
        timm.models.layers.trunc_normal_(self.z_cam_token)
    
    def forward_features(self,x,cam,pos):
        cam_int = (cam - 1).squeeze()
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.features.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.features.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        pos_x = ((pos[:,0] + pos[:,2]) / (2.*self.grid)).type(torch.int64)
        pos_y = ((pos[:,1] + pos[:,3]) / (2.*self.grid)).type(torch.int64)
        x_cam_token = torch.index_select(self.x_cam_token,0,cam_int*self.x_max + pos_x)
        y_cam_token = torch.index_select(self.y_cam_token,0,cam_int*self.y_max + pos_y)
        z_cam_token = torch.index_select(self.z_cam_token,0,cam_int)
        x = torch.cat((x,x_cam_token,y_cam_token,z_cam_token),dim=1)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x = torch.cat((x[:,0],x[:,-3],x[:,-2],x[:,-1]),dim=-1)
        return x

    def forward(self, x, cam, pos):
        output = self.forward_features(x, cam, pos)
        output = output.view(output.size()[0], -1)
        return output

## DeiT-Small + P3DE ##
class FeaturesDeitSmall_distilled_P3DE(nn.Module):
    """
    backbone: DeiT-Small
    """

    def __init__(self,grid):
        super(FeaturesDeitSmall_distilled_P3DE, self).__init__()
        deitsmall = torch.hub.load('facebookresearch/deit:main', 'deit_small_distilled_patch16_224', pretrained=True)
        self.grid = grid
        self.features = deitsmall
        self.num_cam = 9
        self.x_max = int(1920 / self.grid)
        self.y_max = int(1080 / self.grid)
        self.x_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.x_max, 1, deitsmall.embed_dim))
        self.y_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.y_max, 1, deitsmall.embed_dim))
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, deitsmall.embed_dim))
        timm.models.layers.trunc_normal_(self.x_cam_token)
        timm.models.layers.trunc_normal_(self.y_cam_token)
        timm.models.layers.trunc_normal_(self.z_cam_token)
    
    def forward_features(self,x,cam,pos):
        cam_int = (cam - 1).squeeze()
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.features.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.features.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        pos_x = ((pos[:,0] + pos[:,2]) / (2.*self.grid)).type(torch.int64)
        pos_y = ((pos[:,1] + pos[:,3]) / (2.*self.grid)).type(torch.int64)
        x_cam_token = torch.index_select(self.x_cam_token,0,cam_int*self.x_max + pos_x)
        y_cam_token = torch.index_select(self.y_cam_token,0,cam_int*self.y_max + pos_y)
        z_cam_token = torch.index_select(self.z_cam_token,0,cam_int)
        x = torch.cat((x,x_cam_token,y_cam_token,z_cam_token),dim=1)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x = torch.cat((x[:,0],x[:,-3],x[:,-2],x[:,-1]),dim=-1)
        return x

    def forward(self, x, cam, pos):
        output = self.forward_features(x, cam, pos)
        output = output.view(output.size()[0], -1)
        return output

## DeiT-Base + P3DE ##
class FeaturesDeitBase_distilled_P3DE(nn.Module):
    """
    backbone: DeiT-Base
    """

    def __init__(self,grid):
        super(FeaturesDeitBase_distilled_P3DE, self).__init__()
        deitbase = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)
        self.grid = grid
        self.features = deitbase
        self.num_cam = 9
        self.x_max = int(1920 / self.grid)
        self.y_max = int(1080 / self.grid)
        self.x_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.x_max, 1, deitbase.embed_dim))
        self.y_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.y_max, 1, deitbase.embed_dim))
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, deitbase.embed_dim))
        timm.models.layers.trunc_normal_(self.x_cam_token)
        timm.models.layers.trunc_normal_(self.y_cam_token)
        timm.models.layers.trunc_normal_(self.z_cam_token)
    
    def forward_features(self,x,cam,pos):
        cam_int = (cam - 1).squeeze()
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.features.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.features.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        pos_x = ((pos[:,0] + pos[:,2]) / (2.*self.grid)).type(torch.int64)
        pos_y = ((pos[:,1] + pos[:,3]) / (2.*self.grid)).type(torch.int64)
        x_cam_token = torch.index_select(self.x_cam_token,0,cam_int*self.x_max + pos_x)
        y_cam_token = torch.index_select(self.y_cam_token,0,cam_int*self.y_max + pos_y)
        z_cam_token = torch.index_select(self.z_cam_token,0,cam_int)
        x = torch.cat((x,x_cam_token,y_cam_token,z_cam_token),dim=1)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x = torch.cat((x[:,0],x[:,-3],x[:,-2],x[:,-1]),dim=-1)
        return x

    def forward(self, x, cam, pos):
        output = self.forward_features(x, cam, pos)
        output = output.view(output.size()[0], -1)
        return output

class FeaturesRes18(nn.Module):
    """
    backbone: resnet18
    """

    def __init__(self):
        super(FeaturesRes18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        # resnet18.load_state_dict(torch.load('resnet18.pth'))
        modules = list(resnet18.children())[:-1]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        return output


class FeaturesRes50(nn.Module):
    """
    backbone: resnet50
    """

    def __init__(self):
        super(FeaturesRes50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # resnet50.load_state_dict(torch.load('resnet50.pth'))
        modules = list(resnet50.children())[:-1]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        return output


class FeaturesRes101(nn.Module):
    """
    backbone: resnet101
    """

    def __init__(self):
        super(FeaturesRes101, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        # resnet101.load_state_dict(torch.load('resnet101.pth'))
        modules = list(resnet101.children())[:-1]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        return output


class TripletNet(nn.Module):
    """
    Triplet net that can put different backbones
    """

    def __init__(self, config):
        super(TripletNet, self).__init__()
        self.config = config
        self.device = config['device']
        self.features_net = globals()[config['features_net']]()

    def forward_once(self, x):
        """
        input is simply one tensor
        """
        output = self.features_net(x)
        output = output.view(output.size()[0], -1)
        return {'feat': output}

    def compute_distance(self, a_dict, b_dict):
        a_feat = a_dict['feat']
        b_feat = b_dict['feat']
        dist = F.pairwise_distance(a_feat, b_feat)
        return dist

    def forward(self, sample_dict):
        input_anchor = sample_dict['a']
        input_positive = sample_dict['p']
        input_negative = sample_dict['n']

        a_dict = self.forward_once(input_anchor)
        p_dict = self.forward_once(input_positive)
        n_dict = self.forward_once(input_negative)

        output_dict = {
            'ap': self.compute_distance(a_dict, p_dict),
            'an': self.compute_distance(a_dict, n_dict)
        }
        return output_dict

## Using for sie ##
## Using camera ID as a input ##
class TripletNet_cam(nn.Module):
    """
    Triplet net that can put different backbones
    """

    def __init__(self, config):
        super(TripletNet_cam, self).__init__()
        self.config = config
        self.device = config['device']
        self.features_net = globals()[config['features_net']]()

    def forward_once(self, x, cam_index):
        """
        input is simply one tensor
        """
        output = self.features_net(x,cam_index)
        output = output.view(output.size()[0], -1)
        return {'feat': output}

    def compute_distance(self, a_dict, b_dict):
        a_feat = a_dict['feat']
        b_feat = b_dict['feat']
        dist = F.pairwise_distance(a_feat, b_feat)
        return dist

    def forward(self, sample_dict):
        input_anchor = sample_dict['a']
        input_positive = sample_dict['p']
        input_negative = sample_dict['n']

        main_cam = sample_dict['mc']
        sec_cam = sample_dict['sc']

        a_dict = self.forward_once(input_anchor,main_cam)
        p_dict = self.forward_once(input_positive,sec_cam)
        n_dict = self.forward_once(input_negative,sec_cam)

        output_dict = {
            'ap': self.compute_distance(a_dict, p_dict),
            'an': self.compute_distance(a_dict, n_dict)
        }
        return output_dict

## Using for P3DE ##
## Using camera ID and the position of object as inputs ##
class TripletNet_cam_pos(nn.Module):
    """
    Triplet net that can put different backbones
    """

    def __init__(self, config):
        super(TripletNet_cam_pos, self).__init__()
        self.config = config
        self.device = config['device']
        self.sum = config['sum'] if 'sum' in config else False
        self.scale = config['scale'] if 'scale' in config else 1.
        self.features_net = globals()[config['features_net']](config['grid'])

    def forward_once(self, x, cam_index, pos):
        """
        input is simply one tensor
        """
        output = self.features_net(x,cam_index,pos)
        if self.sum:
            cls_token, x_token, y_token, z_token = torch.tensor_split(output,4,dim=-1)
            output = cls_token + self.scale * (x_token + y_token + z_token) / 3.
        output = output.view(output.size()[0], -1)
        return {'feat': output}

    def compute_distance(self, a_dict, b_dict):
        a_feat = a_dict['feat']
        b_feat = b_dict['feat']
        dist = F.pairwise_distance(a_feat, b_feat)
        return dist

    def forward(self, sample_dict):
        input_anchor = sample_dict['a']
        input_positive = sample_dict['p']
        input_negative = sample_dict['n']

        main_cam = sample_dict['mc']
        sec_cam = sample_dict['sc']

        pos_a = sample_dict['pos_a']
        pos_p = sample_dict['pos_p']
        pos_n = sample_dict['pos_n']

        a_dict = self.forward_once(input_anchor,main_cam,pos_a)
        p_dict = self.forward_once(input_positive,sec_cam,pos_p)
        n_dict = self.forward_once(input_negative,sec_cam,pos_n)

        output_dict = {
            'ap': self.compute_distance(a_dict, p_dict),
            'an': self.compute_distance(a_dict, n_dict)
        }
        return output_dict

class TripletLoss(nn.Module):
    """
    Triplet loss, argument: margin
    """

    def __init__(self, config):
        super(TripletLoss, self).__init__()
        self.margin = config['triplet_margin']

    def forward(self, output_dict, sample_dict):
        ap = output_dict['ap']
        an = output_dict['an']
        loss_triplet = torch.mean(F.relu(ap - an + self.margin))
        return loss_triplet


class ASNet(nn.Module):
    """
    ASNet: Appearance Surrounding
    """

    def __init__(self, config):
        super(ASNet, self).__init__()
        self.config = config
        if config['features_net'] == 'FeaturesRes18':
            app = models.resnet18(pretrained=True)
            # app.load_state_dict(torch.load('resnet18.pth'))
            self.app = nn.Sequential(*list(app.children())[:-2])

            sur = models.resnet18(pretrained=True)
            # sur.load_state_dict(torch.load('resnet18.pth'))
            self.sur = nn.Sequential(*list(sur.children())[:-2])

            self.dim = 512
        elif config['features_net'] == 'FeaturesRes50':
            print('model type is FeaturesRes50')
            app = models.resnet50(pretrained=True)
            # app.load_state_dict(torch.load('resnet50.pth'))
            self.app = nn.Sequential(*list(app.children())[:-2])

            sur = models.resnet50(pretrained=True)
            # sur.load_state_dict(torch.load('resnet50.pth'))
            self.sur = nn.Sequential(*list(sur.children())[:-2])

            self.dim = 2048
        elif config['features_net'] == 'FeaturesRes101':
            print('model type is FeaturesRes101')
            app = models.resnet101(pretrained=True)
            # app.load_state_dict(torch.load('resnet101.pth'))
            self.app = nn.Sequential(*list(app.children())[:-2])

            sur = models.resnet101(pretrained=True)
            # sur.load_state_dict(torch.load('resnet101.pth'))
            self.sur = nn.Sequential(*list(sur.children())[:-2])

            self.dim = 2048
        elif config['features_net'] == 'FeaturesDeitTiny':
            print('model type is FeaturesDeitTiny')
            app = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
            self.app = app 

            sur = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
            self.sur = sur

            self.dim = 2048
        elif config['features_net'] == 'FeaturesDeitSmall':
            print('model type is FeaturesDeitSmall')
            app = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
            self.app = app 

            sur = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
            self.sur = sur

            self.dim = 2048
        elif config['features_net'] == 'FeaturesDeitBase':
            print('model type is FeaturesDeitBase')
            app = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
            self.app = app 

            sur = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
            self.sur = sur

            self.dim = 2048
        elif config['features_net'] == 'FeaturesDeitTiny_distilled':
            print('model type is FeaturesDeitTiny_distilled')
            app = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
            self.app = app 

            sur = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
            self.sur = sur

            self.dim = 2048
        else:
            raise NotImplementedError('Currently only implemented for res18')

        assert len(config['zoomout_ratio']) == 1
        self.zoomout_ratio = config['zoomout_ratio'][0]
        self.keep_center = config.get('keep_center', False)

        H = config['cropped_height']
        W = config['cropped_width']

        self.H_size = int(H / self.zoomout_ratio)  # size of center
        self.H_min = int((H - self.H_size) / 2)
        self.H_max = self.H_min + self.H_size

        self.W_size = int(W / self.zoomout_ratio)  # size of center
        self.W_min = int((W - self.W_size) / 2)
        self.W_max = self.W_min + self.W_size

        # print('Crop sizes. H', H, 'H_min', self.H_min, 'H_max', self.H_max, 'W', W, 'W_min', self.W_min, 'W_max', self.W_max)

    def rotate(self, x, angle):
        if angle == 0:
            return x
        elif angle == 90:
            return x.transpose(-2, -1).flip(-2)
        elif angle == 180:
            return x.flip(-2).flip(-1)
        elif angle == 270:
            return x.transpose(-2, -1).flip(-1)
        else:
            raise ValueError('Unable to handle angle ==', angle)

    def forward_once(self, x):
        B, C, H, W = x.size()

        # crop out center
        center = x[:, :, self.H_min:self.H_max, self.W_min:self.W_max].clone()
        surrnd = x

        try:
            ## appearance information resizing -> 224x224
            if self.config['resize']:
                center = torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)(center)
        except:
            pass


        if not self.keep_center:
            surrnd[:, :, self.H_min:self.H_max, self.W_min:self.W_max] = \
                torch.zeros((B, C, self.H_size, self.W_size)).to(x.device)

        # extract features
	## CNN ##
        try:
            center_featmap = self.app(center)
            surrnd_featmap = self.sur(surrnd)
            # average pooling
            center_featvec = F.adaptive_avg_pool2d(center_featmap, output_size=1).view(B, -1)
            surrnd_featvec = F.adaptive_avg_pool2d(surrnd_featmap, output_size=1).view(B, -1)

            return {'center': center_featvec, 'surrnd': surrnd_featvec}
	## Vision Transformer ##
        except:
            tuple_ex = tuple()
            center_featmap = self.app.forward_features(center)
            surrnd_featmap = self.sur.forward_features(surrnd)
            if type(center_featmap) == type(tuple_ex):
                center_featmap = torch.cat(center_featmap,dim=-1)
                surrnd_featmap = torch.cat(surrnd_featmap,dim=-1)
            return {'center': center_featmap, 'surrnd': surrnd_featmap}


    def compute_distance(self, a_dict, b_dict):
        a_center = a_dict['center']
        a_surrnd = a_dict['surrnd']
        b_center = b_dict['center']
        b_surrnd = b_dict['surrnd']
        dist_a_center = F.pairwise_distance(a_center, b_center)  # shape (B, 1)
        dist_b_surrnd = F.pairwise_distance(a_surrnd, b_surrnd)
        surrnd_weight = F.cosine_similarity(a_center, b_center)
        dist = (1 - surrnd_weight) * dist_a_center + surrnd_weight * dist_b_surrnd
        return dist

    def forward(self, sample_dict):
        a = sample_dict['a']
        p = sample_dict['p']
        n = sample_dict['n']

        if self.config.get('rotate', False):
            a = self.rotate(a, 90 * np.random.randint(0, 4))

        output_dict = {
            'a': self.forward_once(a),
            'p': self.forward_once(p),
            'n': self.forward_once(n)
        }
        return output_dict


class CooperativeTripletLoss(nn.Module):
    """
    Loss for ASNet
    Cosine similarity as weight
    """

    def __init__(self, config):
        super(CooperativeTripletLoss, self).__init__()
        self.config = config
        self.margin = config.get('triplet_margin', 0.3)
        print('Cooperative Triplet Loss with margin =', self.margin)

    def forward(self, output_dict, sample_dict):
        anc_center = output_dict['a']['center']
        anc_surrnd = output_dict['a']['surrnd']
        pos_center = output_dict['p']['center']
        pos_surrnd = output_dict['p']['surrnd']
        neg_center = output_dict['n']['center']
        neg_surrnd = output_dict['n']['surrnd']

        # compute pos distance
        dist_pos_center = F.pairwise_distance(anc_center, pos_center)  # shape (B, 1)
        dist_pos_surrnd = F.pairwise_distance(anc_surrnd, pos_surrnd)
        pos_surrnd_weight = F.cosine_similarity(anc_center, pos_center)
        dist_pos = (1 - pos_surrnd_weight) * dist_pos_center + pos_surrnd_weight * dist_pos_surrnd

        # compute neg distance
        dist_neg_center = F.pairwise_distance(anc_center, neg_center)
        dist_neg_surrnd = F.pairwise_distance(anc_surrnd, neg_surrnd)
        neg_surrnd_weight = F.cosine_similarity(anc_center, neg_center)
        dist_neg = (1 - neg_surrnd_weight) * dist_neg_center + neg_surrnd_weight * dist_neg_surrnd

        loss_triplet = torch.mean(F.relu(dist_pos - dist_neg + self.margin))
        return loss_triplet
