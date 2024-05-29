import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_scale=2):
        super(UpsampleConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
class UVRelit(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UVRelit, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # Encoder
        self.enc_conv0 = nn.Conv2d(in_channels=16*32*3+self.in_ch, out_channels=64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # New encoder layer
        self.bn4 = nn.BatchNorm2d(1024)
        
        
        # Max pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder with Upsample and Convolution Block
        self.dec_upconv4 = UpsampleConvBlock(1024, 512)  # New decoder upsample layer
        self.dec_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(512)
        self.dec_upconv3 = UpsampleConvBlock(512+512, 256)  # Corrected for concatenated skip connection
        self.dec_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(256)
        self.dec_upconv2 = UpsampleConvBlock(256 + 256, 128)
        self.dec_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_upconv1 = UpsampleConvBlock(128 + 128, 64)
        self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        
        # Final output
        self.final_conv = nn.Conv2d(64, self.out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.bn0(self.enc_conv0(x)))
        p0 = self.pool(e0)
        e1 = F.relu(self.bn1(self.enc_conv1(p0)))
        p1 = self.pool(e1)
        e2 = F.relu(self.bn2(self.enc_conv2(p1)))
        p2 = self.pool(e2)
        e3 = F.relu(self.bn3(self.enc_conv3(p2)))
        p3 = self.pool(e3)
        e4 = F.relu(self.bn4(self.enc_conv4(p3)))  # New encoder output
        p4 = self.pool(e4)
        
        # Decoder with skip connections
        # d3 = self.dec_upconv3(e3)
        d4 = self.dec_upconv4(e4)  # Process new encoder layer
        d4 = F.relu(self.dec_bn4(self.dec_conv4(d4)))
        d3 = self.dec_upconv3(torch.cat([d4, e3], dim=1))
        d3 = F.relu(self.dec_bn3(self.dec_conv3(d3)))
        d2 = self.dec_upconv2(torch.cat([d3, e2], dim=1))
        d2 = F.relu(self.dec_bn2(self.dec_conv2(d2)))
        d1 = self.dec_upconv1(torch.cat([d2, e1], dim=1))
        d1 = F.relu(self.dec_bn1(self.dec_conv1(d1)))
        
        # Final output
        out = self.final_conv(d1)
        
        return out
# This is with 5.1M parameters
# class UVRelit(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(UVRelit, self).__init__()
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         # Encoder
#         self.enc_conv0 = nn.Conv2d(in_channels=16*32*3+self.in_ch, out_channels=64, kernel_size=3, padding=1)
#         self.bn0 = nn.BatchNorm2d(64)
#         self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(512)
        
#         # Max pool
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Decoder with Upsample and Convolution Block
#         self.dec_upconv3 = UpsampleConvBlock(512, 256)  # Corrected for concatenated skip connection
#         self.dec_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.dec_bn3 = nn.BatchNorm2d(256)
#         self.dec_upconv2 = UpsampleConvBlock(256 + 256, 128)
#         self.dec_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.dec_bn2 = nn.BatchNorm2d(128)
#         self.dec_upconv1 = UpsampleConvBlock(128 + 128, 64)
#         self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.dec_bn1 = nn.BatchNorm2d(64)
        
#         # Final output
#         self.final_conv = nn.Conv2d(64, self.out_ch, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         e0 = F.relu(self.bn0(self.enc_conv0(x)))
#         p0 = self.pool(e0)
#         e1 = F.relu(self.bn1(self.enc_conv1(p0)))
#         p1 = self.pool(e1)
#         e2 = F.relu(self.bn2(self.enc_conv2(p1)))
#         p2 = self.pool(e2)
#         e3 = F.relu(self.bn3(self.enc_conv3(p2)))
#         p3 = self.pool(e3)
        
#         # Decoder with skip connections
#         d3 = self.dec_upconv3(e3)
#         d3 = F.relu(self.dec_bn3(self.dec_conv3(d3)))
#         d2 = self.dec_upconv2(torch.cat([d3, e2], dim=1))
#         d2 = F.relu(self.dec_bn2(self.dec_conv2(d2)))
#         d1 = self.dec_upconv1(torch.cat([d2, e1], dim=1))
#         d1 = F.relu(self.dec_bn1(self.dec_conv1(d1)))
        
#         # Final output
#         out = self.final_conv(d1)
        
#         return out

class EnvMapEncoder(nn.Module):
    def __init__(self,map_size=16,latent_size=4):
        super(EnvMapEncoder, self).__init__()
        # Load pretrained VGG19 model
        # vgg19 = models.vgg19(pretrained=True)
        efficientnet_b1 = models.efficientnet_b1(pretrained=True)
        # Remove classifier layers and keep only the feature extractor part
        # self.features = vgg19.features
        self.features = nn.Sequential(*list(efficientnet_b1.children())[:-2])
        
        # Define a custom additional layers to adjust the output size to 4x16x16
        # self.additional_layers = nn.Sequential(
        #     nn.Conv2d(512, latent_size, kernel_size=1),  # Reduces channel size to 4
        #     nn.AdaptiveMaxPool2d((map_size, map_size))     # Resize feature maps to 16x16
        # )
        self.additional_layers = nn.Sequential(
            nn.Conv2d(1280, latent_size, kernel_size=1),  # Reduces channel size to 4, note channel count change
            nn.AdaptiveMaxPool2d((map_size, map_size))    # Resize feature maps to 16x16
        )
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        # Transform feature maps to the desired output size
        x = self.additional_layers(x)
        return x

class Encoder_Unet(nn.Module):
    def __init__(self, in_ch=48):
        super(Encoder_Unet, self).__init__()
        # Assuming in_ch is the initial channel size for the encoder input
        self.enc_conv0 = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # Added new layer
        self.bn4 = nn.BatchNorm2d(1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        e0 = F.relu(self.bn0(self.enc_conv0(x)))
        p0 = self.pool(e0)
        e1 = F.relu(self.bn1(self.enc_conv1(p0)))
        p1 = self.pool(e1)
        e2 = F.relu(self.bn2(self.enc_conv2(p1)))
        p2 = self.pool(e2)
        e3 = F.relu(self.bn3(self.enc_conv3(p2)))
        e4 = F.relu(self.bn4(self.enc_conv4(e3)))  # Added new layer processing
        p3 = self.pool(e4)
        
        # Return list of intermediate features and final output
        return [p0, e1, e2, e3, e4], p3

class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_scale=2):
        super(UpsampleConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
    


class Decoder_Unet(nn.Module):
    def __init__(self, in_ch=1028,out_ch=48):
        super(Decoder_Unet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # Decoder Layers
        self.decoder_layers = nn.ModuleList([
            UpsampleConvBlock(self.in_ch, 512),        # Layer 4
            nn.Sequential(
                UpsampleConvBlock(1024, 256),   # Layer 3 (512 from skip connection)
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256)
            ),
            nn.Sequential(
                UpsampleConvBlock(512, 128),   # Layer 2 (512 from skip connection)
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128)
            ),
            nn.Sequential(
                UpsampleConvBlock(256, 64),    # Layer 1 (256 from skip connection)
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)
            )
        ])
        self.final_conv = nn.Conv2d(64, self.out_ch, kernel_size=1)

        # self.apply(self.init_weights)


    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, encoded_layers, bottleneck):
        
        d = self.decoder_layers[0](bottleneck)
        for i, layer in enumerate(self.decoder_layers[1:]):
            d = torch.cat([d, encoded_layers[-(i + 2)]], dim=1)  # Concatenate in reverse order
            d = layer(d)
        out = self.final_conv(d)
        return out


class LatentUnet(nn.Module):
    def __init__(self,in_ch=3,out_ch=3):
        super(LatentUnet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.encoder = Encoder_Unet(in_ch=in_ch)
        self.decoder = Decoder_Unet(in_ch=1028,out_ch=out_ch)
        self.envmap_encoder = EnvMapEncoder(map_size=16,latent_size=4)

    def forward(self,image,envmap):
        latent = self.envmap_encoder(envmap)
        features,bottleneck = self.encoder(image)
        new_bottleneck = torch.cat((bottleneck,latent),dim=1)
        relit = self.decoder(features,new_bottleneck)
        return relit

def format_parameters(num_params):
    if num_params < 1e6:
        return str(num_params)
    elif num_params < 1e9:
        return "{:.1f} million".format(num_params / 1e6)
    else:
        return "{:.1f} billion".format(num_params / 1e9)


def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return format_parameters(num_params)

if __name__ == "__main__":
    # relit = UVRelit()
    latent = LatentUnet()
    
    # print(output.shape)
    print(count_parameters(latent))


