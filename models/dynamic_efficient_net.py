import torchvision
import torchvision.transforms as transforms
from torch import nn
from .blocks import MBConv, FusedMBConv, ConvBatchActivation

class DynamicEfficientNet(nn.Module):
    def __init__(self, stage_configs, num_classes=1):
        super().__init__()
        
        # 1. Stem
        self.stem = ConvBatchActivation(3, 32, kernel_size=3, stride=2)
        stages = []
        in_channels = 32 
        
        for config in stage_configs:
            stage_layers = []
            #Parâmetros fixos
            out_channels = config['out_channels']
            num_layers = config['num_layers']
            stride = config['stride']
            #Parâmetros dinâmicos
            block_type = config['block_type']
            kernel_size = config['kernel_size']
            expand_ratio = config['expand_ratio']
            attention_type = config['attention_type'] 
            att_reduction = config['att_reduction']

            for i in range(num_layers):
                current_stride = stride if i == 0 else 1
                
                if block_type == 'MBConv':
                    block = MBConv(in_channels, out_channels, kernel_size, 
                                   current_stride, expand_ratio, attention_type, att_reduction)
                elif block_type == 'FusedMBConv':
                    block = FusedMBConv(in_channels, out_channels, kernel_size, 
                                        current_stride, expand_ratio, attention_type, att_reduction)
                #TODO: Adicionar bloco Residual, ConveNext
                else:
                    raise ValueError(f"Tipo de bloco desconhecido: {block_type}")
                
                stage_layers.append(block)
                in_channels = out_channels 
            
            stages.append(nn.Sequential(*stage_layers))
            
        self.stages = nn.Sequential(*stages)
        
        # 3.Head
        last_conv_channels = 1280
        self.pre_head = ConvBatchActivation(in_channels, last_conv_channels, kernel_size=1, stride=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(last_conv_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pre_head(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x