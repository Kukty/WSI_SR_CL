from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import torch.nn as nn
# from basicsr.archs.arch_util import make_layer,RRDB,pixel_unshuffle
from arch_util import make_layer,RRDB,pixel_unshuffle
from timm.models.layers import trunc_normal_
from torch.nn import functional as F
import torchvision.models.resnet as resnet 
def make_resnet_layer(
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1

        if stride != 1:
            downsample = nn.Sequential(
                resnet.conv1x1(planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                planes, planes, stride, downsample, 1, 64, previous_dilation, norm_layer
            )
        )
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    groups=1,
                    base_width=64,
                    dilation=1,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
# class 

class models_esrc(nn.Module):
    def __init__(self, num_in_ch,num_out_ch=3, scale=4, num_feat=64, num_block=23,
                 drop_rate=0.3, num_grow_ch=32,num_classes=5,head_init_scale=1,input_size=224):
        super(models_esrc, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # -----------------upsampler-------------------------------------
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #----------------------Classification head ----------------------
        self.layer1 = make_resnet_layer(resnet.Bottleneck,num_feat,2,stride=2,dilate=False)
        self.layer2 = make_resnet_layer(resnet.Bottleneck,256,2,stride=2,dilate=False)
        self.dropout = nn.Dropout(p=drop_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        
        # self.norm = nn.LayerNorm(num_feat, eps=1e-6) # final norm layer
        
        
        # hidden_size = 512
        # self.flatten = nn.Flatten()

        #### last version
        self.head = nn.Sequential(
                    nn.BatchNorm1d(num_feat),
                    nn.Linear(num_feat, num_feat // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(num_feat // 2, num_classes)
                )
        ####


        # self.head = nn.Linear(num_feat, num_classes)
        # self.dropout = nn.Dropout(p=drop_rate)
        # self.apply(self._init_weights)
        # self.head[1].weight.data.mul_(head_init_scale)
        # self.head[1].bias.data.mul_(head_init_scale)
        # self.head[3].weight.data.mul_(head_init_scale)
        # self.head[3].bias.data.mul_(head_init_scale)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward_features(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        return feat 

    def forward_class_head(self,feat):
        ######
        # x = feat.mean([-2, -1])
        # logits = self.head(x)
        #  # global average pooling, (N, C, H, W) -> (N, C)
        ######
        # x = self.dropout(x)
        # # x = self.flatten(feat)
        
        x = self.layer1(feat)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
    
    def forward_upsampler(self,feat):
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def forward(self, x,mode=None):
        feat = self.forward_features(x)
        
        if mode == 'cl':
            logits = self.forward_class_head(feat)
            return logits
        if mode == 'sr':
            out =self.forward_upsampler(feat)
            return out 
        else:
            logits = self.forward_class_head(feat)
            out =self.forward_upsampler(feat)
            return logits,out