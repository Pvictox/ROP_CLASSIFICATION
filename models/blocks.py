import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch

# Essa classe é um helper que combina uma conv2d, batchnorm e uma ativação em uma única camada sequencial
class ConvBatchActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        padding = (kernel_size - 1) // 2  # Mantém o tamanho espacial
        super(ConvBatchActivation, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) #TODO: Outras?
        )

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4): # Padrão 4 (se_ratio=0.25)
        super().__init__()
        reduced_dim = max(1, in_channels // reduction_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x)

#Usado pelo CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_c, reduction_ratio=16):
        super().__init__()
        reduced_dim = max(1, in_c // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_c, reduced_dim, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(reduced_dim, in_c, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_c, reduction_ratio=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_c, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)
        
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# B MBConv do EfficientNet original
class MBConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, expand_ratio, 
                 attention_type='se', att_reduction=4):
        super().__init__()
        self.use_residual = (in_c == out_c) and (stride == 1)
        hidden_dim = int(in_c * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBatchActivation(in_c, hidden_dim, kernel_size=1, stride=1))
        
        layers.append(ConvBatchActivation(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                                stride=stride, groups=hidden_dim))
        if attention_type == 'se':
            layers.append(SEModule(hidden_dim, reduction_ratio=att_reduction))
        elif attention_type == 'cbam':
            layers.append(CBAM(hidden_dim, reduction_ratio=att_reduction))
        
        layers.append(nn.Conv2d(hidden_dim, out_c, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_c))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)

# Fused-MBConv (do EfficientNetV2)
class FusedMBConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, expand_ratio, 
                 attention_type='se', att_reduction=4):
        super().__init__()
        self.use_residual = (in_c == out_c) and (stride == 1)
        hidden_dim = int(in_c * expand_ratio)

        layers = []
    
        if expand_ratio != 1:
            layers.append(ConvBatchActivation(in_c, hidden_dim, kernel_size=kernel_size, stride=stride))
    
        if expand_ratio != 1 and attention_type != 'none':
            if attention_type == 'se':
                layers.append(SEModule(hidden_dim, reduction_ratio=att_reduction))
            elif attention_type == 'cbam':
                layers.append(CBAM(hidden_dim, reduction_ratio=att_reduction))
        
        projection_in = hidden_dim if expand_ratio != 1 else in_c
        projection_stride = stride if expand_ratio == 1 else 1
        layers.append(nn.Conv2d(projection_in, out_c, 1, stride=projection_stride, bias=False))
        layers.append(nn.BatchNorm2d(out_c))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)