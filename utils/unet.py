import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self):
        super(UVRelit, self).__init__()

        # Encoder
        self.enc_conv0 = nn.Conv2d(in_channels=16*32*3+3, out_channels=64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        
        # Max pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder with Upsample and Convolution Block
        self.dec_upconv3 = UpsampleConvBlock(512, 256)  # Corrected for concatenated skip connection
        self.dec_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(256)
        self.dec_upconv2 = UpsampleConvBlock(256 + 256, 128)
        self.dec_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_upconv1 = UpsampleConvBlock(128 + 128, 64)
        self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        
        # Final output
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

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
        
        # Decoder with skip connections
        d3 = self.dec_upconv3(e3)
        d3 = F.relu(self.dec_bn3(self.dec_conv3(d3)))
        d2 = self.dec_upconv2(torch.cat([d3, e2], dim=1))
        d2 = F.relu(self.dec_bn2(self.dec_conv2(d2)))
        d1 = self.dec_upconv1(torch.cat([d2, e1], dim=1))
        d1 = F.relu(self.dec_bn1(self.dec_conv1(d1)))
        
        # Final output
        out = self.final_conv(d1)
        
        return out

class Encoder(nn.Module):
    def __init__(self, in_ch=48):
        super(Encoder, self).__init__()
        # Assuming in_ch is the initial channel size for the encoder input
        self.enc_conv0 = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e0 = F.relu(self.bn0(self.enc_conv0(x)))
        p0 = self.pool(e0)
        e1 = F.relu(self.bn1(self.enc_conv1(p0)))
        p1 = self.pool(e1)
        e2 = F.relu(self.bn2(self.enc_conv2(p1)))
        p2 = self.pool(e2)
        e3 = F.relu(self.bn3(self.enc_conv3(p2)))
        p3 = self.pool(e3)
        
        # Return list of intermediate features and final output
        return [p0, e1, e2, e3], e3

class Decoder(nn.Module):
    def __init__(self, out_ch=48):
        super(Decoder, self).__init__()
        # Decoder Layers
        self.dec_upconv3 = UpsampleConvBlock(512, 256)  # Corrected for concatenated skip connection
        self.dec_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(256)
        self.dec_upconv2 = UpsampleConvBlock(256 + 256, 128)
        self.dec_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_upconv1 = UpsampleConvBlock(128 + 128, 64)
        self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, encoded_layers, bottleneck):
        d3 = self.dec_upconv3(bottleneck)
        d3 = F.relu(self.dec_bn3(self.dec_conv3(d3)))
        d2 = self.dec_upconv2(torch.cat([d3, encoded_layers[2]], dim=1))
        d2 = F.relu(self.dec_bn2(self.dec_conv2(d2)))
        d1 = self.dec_upconv1(torch.cat([d2, encoded_layers[1]], dim=1))
        d1 = F.relu(self.dec_bn1(self.dec_conv1(d1)))
        
        out = self.final_conv(d1)
        return out


# class Encoder(nn.Module):
#     def __init__(self, in_channels=3, features=[64, 128, 256, 512, 1024]):
#         super(Encoder, self).__init__()
#         self.layers = nn.ModuleList()

#         # Initial layer
#         self.layers.append(
#             nn.Sequential(
#                 nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
#                 nn.BatchNorm2d(features[0]),
#                 nn.ReLU(inplace=True)
#             )
#         )

#         # Subsequent layers
#         for feature_in, feature_out in zip(features[:-1], features[1:]):
#             self.layers.append(
#                 nn.Sequential(
#                     nn.MaxPool2d(kernel_size=2, stride=2),
#                     nn.Conv2d(feature_in, feature_out, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(feature_out),
#                     nn.ReLU(inplace=True),
#                 )
#             )

#     def forward(self, x):
#         features = []
#         for layer in self.layers[:-1]:
#             x = layer(x)
#             features.append(x)
#         bottleneck = self.layers[-1](x)
#         return features, bottleneck

# class Decoder(nn.Module):
#     def __init__(self, out_channels=3, features=[1024, 512, 256, 128, 64]):
#         super(Decoder, self).__init__()
#         self.upconvs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.dec_convs = nn.ModuleList()

#         # Initialize up-sampling and convolution layers
#         for feature_in, feature_out in zip(features[:-1], features[1:]):
#             self.upconvs.append(
#                 nn.ConvTranspose2d(feature_in, feature_out, kernel_size=2, stride=2)
#             )
#             self.dec_convs.append(
#                 nn.Sequential(
#                     nn.Conv2d(feature_in // 2 + feature_out, feature_out, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(feature_out),
#                     nn.ReLU(inplace=True),
#                 )
#             )

#         # Final layer
#         self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)

#     def forward(self, features, bottleneck):
#         x = bottleneck
#         for upconv, dec_conv, skip_feature in zip(self.upconvs, self.dec_convs, features[::-1][1:]):
#             x = upconv(x)
#             x = torch.cat([x, skip_feature], dim=1)
#             x = dec_conv(x)
#         x = self.final_conv(x)
#         return x

# class Encoder(nn.Module):
#     def __init__(self, in_ch=48):
#         super(Encoder, self).__init__()
#         # Assuming in_ch is the initial channel size for the encoder input
#         self.enc_conv0 = nn.Conv2d(in_channels=16*32*3+in_ch, out_channels=64, kernel_size=3, padding=1)
#         self.bn0 = nn.BatchNorm2d(64)
#         self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(512)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         e0 = F.relu(self.bn0(self.enc_conv0(x)))
#         p0 = self.pool(e0)
#         e1 = F.relu(self.bn1(self.enc_conv1(p0)))
#         p1 = self.pool(e1)
#         e2 = F.relu(self.bn2(self.enc_conv2(p1)))
#         p2 = self.pool(e2)
#         e3 = F.relu(self.bn3(self.enc_conv3(p2)))
        
#         return [p0, e1, e2, e3], e3

# class Encoder(nn.Module):
#     def __init__(self, in_ch=48):
#         super(Encoder, self).__init__()
#         # Assuming in_ch is the initial channel size for the encoder input
#         self.enc_conv0 = nn.Conv2d(in_channels=16*32*3+in_ch, out_channels=64, kernel_size=3, padding=1)
#         self.bn0 = nn.BatchNorm2d(64)
#         self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(512)
#         self.enc_conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # Added new layer
#         self.bn4 = nn.BatchNorm2d(1024)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         e0 = F.relu(self.bn0(self.enc_conv0(x)))
#         p0 = self.pool(e0)
#         e1 = F.relu(self.bn1(self.enc_conv1(p0)))
#         p1 = self.pool(e1)
#         e2 = F.relu(self.bn2(self.enc_conv2(p1)))
#         p2 = self.pool(e2)
#         e3 = F.relu(self.bn3(self.enc_conv3(p2)))
#         e4 = F.relu(self.bn4(self.enc_conv4(e3)))  # Added new layer processing
#         p3 = self.pool(e4)
        
#         # Return list of intermediate features and final output
#         return [p0, e1, e2, e3, e4], p3


# class Decoder(nn.Module):
#     def __init__(self, out_ch=48):
#         super(Decoder, self).__init__()
#         # Decoder Layers
#         self.dec_upconv3 = UpsampleConvBlock(512, 256)  # Corrected for concatenated skip connection
#         self.dec_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.dec_bn3 = nn.BatchNorm2d(256)
#         self.dec_upconv2 = UpsampleConvBlock(256 + 256, 128)
#         self.dec_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.dec_bn2 = nn.BatchNorm2d(128)
#         self.dec_upconv1 = UpsampleConvBlock(128 + 128, 64)
#         self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.dec_bn1 = nn.BatchNorm2d(64)
#         self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

#     def forward(self, encoded_layers, bottleneck):
#         d3 = self.dec_upconv3(bottleneck)
#         d3 = F.relu(self.dec_bn3(self.dec_conv3(d3)))
#         d2 = self.dec_upconv2(torch.cat([d3, encoded_layers[2]], dim=1))
#         d2 = F.relu(self.dec_bn2(self.dec_conv2(d2)))
#         d1 = self.dec_upconv1(torch.cat([d2, encoded_layers[1]], dim=1))
#         d1 = F.relu(self.dec_bn1(self.dec_conv1(d1)))
        
#         out = self.final_conv(d1)
#         return out
    
# class Decoder(nn.Module):
#     def __init__(self, out_ch=48):
#         super(Decoder, self).__init__()
#         # Decoder Layers
#         self.dec_upconv4 = UpsampleConvBlock(1024, 512)  # Assuming Encoder returns 5 layers
#         self.dec_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.dec_bn4 = nn.BatchNorm2d(512)
#         self.dec_upconv3 = UpsampleConvBlock(512 + 512, 256)  # Skip connection from encoder
#         self.dec_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.dec_bn3 = nn.BatchNorm2d(256)
#         self.dec_upconv2 = UpsampleConvBlock(256 + 256, 128)  # Skip connection from encoder
#         self.dec_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.dec_bn2 = nn.BatchNorm2d(128)
#         self.dec_upconv1 = UpsampleConvBlock(128 + 128, 64)  # Skip connection from encoder
#         self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.dec_bn1 = nn.BatchNorm2d(64)
#         self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

#     def forward(self, encoded_layers, bottleneck):
#         d4 = self.dec_upconv4(bottleneck)  # Added new layer processing
#         d4 = F.relu(self.dec_bn4(self.dec_conv4(d4)))
#         d3 = self.dec_upconv3(torch.cat([d4, encoded_layers[3]], dim=1))  # Concatenate with skip connection
#         d3 = F.relu(self.dec_bn3(self.dec_conv3(d3)))
#         d2 = self.dec_upconv2(torch.cat([d3, encoded_layers[2]], dim=1))  # Concatenate with skip connection
#         d2 = F.relu(self.dec_bn2(self.dec_conv2(d2)))
#         d1 = self.dec_upconv1(torch.cat([d2, encoded_layers[1]], dim=1))  # Concatenate with skip connection
#         d1 = F.relu(self.dec_bn1(self.dec_conv1(d1)))
        
#         out = self.final_conv(d1)
#         return out
# class Decoder(nn.Module):
#     def __init__(self, out_ch=48):
#         super(Decoder, self).__init__()
#         # Decoder Layers
#         self.decoder_layers = nn.ModuleList([
#             UpsampleConvBlock(1024, 512),        # Layer 4
#             nn.Sequential(
#                 UpsampleConvBlock(1024, 256),   # Layer 3 (512 from skip connection)
#                 nn.Conv2d(256, 256, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(256)
#             ),
#             nn.Sequential(
#                 UpsampleConvBlock(512, 128),   # Layer 2 (512 from skip connection)
#                 nn.Conv2d(128, 128, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(128)
#             ),
#             nn.Sequential(
#                 UpsampleConvBlock(256, 64),    # Layer 1 (256 from skip connection)
#                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(64)
#             )
#         ])
#         self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

#     def forward(self, encoded_layers, bottleneck):
#         d = self.decoder_layers[0](bottleneck)
#         for i, layer in enumerate(self.decoder_layers[1:]):
#             d = torch.cat([d, encoded_layers[-(i + 2)]], dim=1)  # Concatenate in reverse order
#             d = layer(d)
#         out = self.final_conv(d)
#         return out
class PosEnc(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        super(PosEnc, self).__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)

    def forward(self, x):
        C, H, W = x.shape
        x = x.permute(1,2,0).reshape(-1, C)  # Reshape to (B*H*W, C)
        out = []
        if self.include_input:
            out.append(x)
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        encoded = torch.cat(out, -1)  # Shape becomes (B*H*W, C_encoded)
        C_encoded = encoded.shape[1]
        return encoded.view(H, W, C_encoded).permute(2,1,0)  # Back to (B, C_encoded, H, W)


if __name__ == "__main__":

    gauss = torch.rand(1,59,256,256) # all features
    encoder = Encoder(in_ch=59)
    decoder = Decoder(out_ch=59)
    features,bottleneck = encoder(gauss)
    print(bottleneck.shape)
    output = decoder(features,bottleneck)
    print(output.shape)