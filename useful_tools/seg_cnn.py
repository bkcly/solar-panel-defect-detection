import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.seq(inputs)

class Block(nn.Module):
    def __init__(self, src_channels, dst_channels):
        super().__init__()
        self.seq1 = ConvBNAct(src_channels, dst_channels)
        self.seq2 = ConvBNAct(dst_channels, dst_channels)
        self.seq3 = ConvBNAct(dst_channels, dst_channels)

    def forward(self, x):
        result = self.seq1(x)
        result = self.seq2(result)
        result = self.seq3(result)
        return result

class UNetUp(nn.Module):
    def __init__(self, down_channels, right_channels):
        super().__init__()
        self.conv = nn.Conv2d(down_channels, right_channels, kernel_size=1, stride=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, left, bottom):
        from_bottom = self.up(bottom)
        from_bottom = self.conv(from_bottom)
        result = torch.cat([left, from_bottom], 1)
        return result

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), dilation=2, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), dilation=2, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.conv2(self.relu(out))
        out = self.bn2(out)
        return torch.cat((x, self.relu2(out)), dim=1)

class SegmentationCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Get encoder from VGG16
        vgg = vgg16_bn(weights='DEFAULT')
        self.encoder_blocks = []
        cur_block = nn.Sequential()
        layers_last_module_names = ['5', '12', '22', '32', '42']
        
        for name, child in vgg.features.named_children():
            cur_block.add_module(name, child)
            if name in layers_last_module_names:
                self.encoder_blocks.append(cur_block)
                cur_block = nn.Sequential()
        
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        encoder_channels = [64, 128, 256, 512, 512]
        
        # Bottleneck - outputs 1024 channels (512 from input + 512 from bottleneck)
        self.bottle = Bottleneck(512, 512)
        
        # Decoder blocks - first block handles 1024 channels from bottleneck
        self.blocks = nn.ModuleList()
        self.blocks.append(Block(1024, 1024))  # Changed from 512 to 1024
        
        # Upsampling blocks
        self.ups = nn.ModuleList()
        decoder_channels = [512, 256, 128, 64]  # Channels after each upsampling
        
        # First upsampling block (special case)
        self.ups.append(UNetUp(1024, 512))  # Input: 1024 channels from first decoder block
        self.blocks.append(Block(1024, 512))  # 512 from encoder skip + 512 from upsampling
        
        # Remaining upsampling blocks
        for i in range(1, len(decoder_channels) - 1):
            in_channels = decoder_channels[i-1]
            out_channels = decoder_channels[i]
            self.ups.append(UNetUp(in_channels, out_channels))
            self.blocks.append(Block(out_channels * 2, out_channels))  # Double channels due to skip connection
        
        # Final upsampling and block
        self.ups.append(UNetUp(128, 64))
        self.blocks.append(Block(128, 64))  # 64 from encoder skip + 64 from upsampling
        
        # Final convolution
        self.last_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)
        
        # Bottleneck - outputs 1024 channels
        x = self.bottle(encoder_outputs[-1])  # 512 -> 1024 channels
        
        # First decoder block
        x = self.blocks[0](x)  # Process 1024 channels
        
        # Remaining decoder blocks with skip connections
        for i in range(len(self.encoder_blocks) - 1):
            encoder_output = encoder_outputs[len(self.encoder_blocks) - 2 - i]
            x = self.ups[i](encoder_output, x)
            x = self.blocks[i + 1](x)
        
        # Final convolution
        x = self.last_conv(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        # Handle DataParallel prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # Remove 'module.' prefix
            new_state_dict[k] = v
        return super().load_state_dict(new_state_dict, strict=strict)
