import torch
import torch.nn as nn
import torch.nn.functional as F

# from loss_utils import IDMRFLoss
import numpy as np
from imageio.v2 import imread,imwrite

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

class EnvMapEncoder(nn.Module):
    def __init__(self):
        super(EnvMapEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(256, 256)  # Adjust the input size based on your specific image dimensions

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x.T)
        return x.unsqueeze(0)

# Decoder using Transposed Convolutions

class Decoder_shape(nn.Module):
    def __init__(self,latent=256,gaussians=512,shape_len=10):
        super(Decoder_shape, self).__init__()
        self.latent = latent
        self.gaussians = gaussians
        self.relu = nn.ReLU()
        self.shape_len = shape_len
        self.fc = nn.Linear(self.latent, 128 * (self.gaussians//16) * (self.gaussians//16))
        
        # Adding more layers and adjusting parameters to achieve the desired output shape
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(16, self.shape_len, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        
        x = self.relu((self.fc(x)))
        x = x.view(x.size(0), 128, (self.gaussians//16), (self.gaussians//16))
        x = self.relu((self.deconv1(x)))
        x = self.relu((self.deconv2(x)))
        x = self.relu((self.deconv3(x)))
        x = self.relu((self.deconv4(x)))
        x = self.deconv5(x)
        
        x=x.reshape(self.shape_len,self.gaussians*self.gaussians)
        pos_delta=x[0:3,:]
        scales_delta=x[3:6,:]
        rotations_delta=x[6:10,:]
        opacity_delta=x[10:11,:]
        return pos_delta,scales_delta,rotations_delta,opacity_delta
    
class Decoder_alpha(nn.Module):
    def __init__(self,latent=256,gaussians=512):
        super(Decoder_alpha, self).__init__()
        self.relu = nn.ReLU()
        self.latent = latent
        self.gaussians = gaussians

        
        self.fc = nn.Linear(self.latent, 128 * (self.gaussians//16) * (self.gaussians//16))
        
        # Adding more layers and adjusting parameters to achieve the desired output shape
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu((self.fc(x)))
        x = x.view(x.size(0), 128, (self.gaussians//16), (self.gaussians//16))
        x = self.relu((self.deconv1(x)))
        x = self.relu((self.deconv2(x)))
        x = self.relu((self.deconv3(x)))
        x = self.relu((self.deconv4(x)))
        x = self.deconv5(x)
        opacity=x.reshape(1,self.gaussians*self.gaussians) # 1,N
        
        
        return opacity

class Decoder_RGB(nn.Module):
    def __init__(self,latent=256,gaussians=512,view=3,color=3):
        super(Decoder_RGB, self).__init__()
        # Define the initial fully connected layer
        
        self.gaussians = gaussians
        self.latent = latent
        self.color = color
        self.view = view
        
        self.relu = nn.ReLU()
        self.fc = nn.Linear(latent+self.view, 256 * (self.gaussians//16) * (self.gaussians//16))  # Increase from 128 to 256
        
        # Define the transposed convolutional layers
        # The sequence is designed to double the spatial dimensions step by step
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 64x64
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)   # Output: 128x128
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)    # Output: 256x256
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)    # Output: 512x512
        self.deconv5 = nn.ConvTranspose2d(16, self.color, kernel_size=3, stride=1, padding=1)                      # Output: 512x512


    def forward(self, latent_code,view):
        x = torch.cat((latent_code,view),dim=2)
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 256, (self.gaussians//16), (self.gaussians//16))
        x = self.relu((self.deconv1(x)))
        x = self.relu((self.deconv2(x)))
        x = self.relu((self.deconv3(x)))
        x = self.relu((self.deconv4(x)))
        x = self.deconv5(x)
        x=x.reshape(self.color,self.gaussians*self.gaussians)
        # x=x.T

        return x
class Decoder_RGB_SH(nn.Module):
    def __init__(self, latent=256, gaussians=256, sh_degree=3):
        super(Decoder_RGB_SH, self).__init__()
        
        self.gaussians = gaussians
        self.latent = latent
        self.color = (sh_degree + 1) ** 2
        if sh_degree == 3:
            self.factor = 64
        else: 
            self.factor = 16
        # self.factor = pow(2,sh_degree) * 8
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(latent, 256 * self.factor * self.factor)  
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)   
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)    
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)    
        self.deconv5 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)  # Changed output channels to 3                      
        
    def forward(self, latent_code):
        x = self.relu(self.fc(latent_code))
        x = x.view(x.size(0), 256, self.factor, self.factor)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.deconv5(x)
        
        return x.view(-1, self.color, 3, self.gaussians * self.gaussians).permute(0, 1, 3, 2).reshape(-1, self.color, 3)
    
class Decoder_RGB_SH_Big(nn.Module):
    def __init__(self, latent=256, gaussians=256, sh_degree=3):
        super(Decoder_RGB_SH_Big, self).__init__()
        
        self.gaussians = gaussians
        self.latent = latent
        self.color = (sh_degree + 1) ** 2
        self.factor = 32  # Adjusted factor size for the desired output shape
        
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(latent, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 512 * self.factor * self.factor)  # Adjusted output size for upconv layers
        
        # Upconvolutional layers
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Output layer
        self.output_layer = nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1)
        
    def forward(self, latent_code):
        x = self.relu(self.fc1(latent_code))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        
        x = x.view(x.size(0), 512, self.factor, self.factor)
        
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.upconv3(x))
        x = self.output_layer(x)
        
        return x.view(-1, self.color, 3, self.gaussians * self.gaussians).permute(0, 1, 3, 2).reshape(-1, self.color, 3)
    
class Decoder_Shape_SH_Big(nn.Module):
    def __init__(self, latent=256, gaussians=256, shape_len=10):
        super(Decoder_Shape_SH_Big, self).__init__()
        
        self.gaussians = gaussians
        self.latent = latent
        self.latent = latent
        self.gaussians = gaussians
        self.relu = nn.ReLU()
        self.shape_len = shape_len
        self.factor = 32  # Adjusted factor size for the desired output shape
        
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(latent, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 512 * self.factor * self.factor)  # Adjusted output size for upconv layers
        
        # Upconvolutional layers
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Output layer
        self.output_layer = nn.Conv2d(64, 10, kernel_size=3, stride=1, padding=1)
        
    def forward(self, latent_code):
        x = self.relu(self.fc1(latent_code))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        
        x = x.view(x.size(0), 512, self.factor, self.factor)
        
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.upconv3(x))
        x = self.output_layer(x)
        x=x.reshape(self.shape_len,self.gaussians*self.gaussians)
        
        pos_delta=x[0:3,:]
        scales_delta=x[3:6,:]
        rotations_delta=x[6:10,:]
        # opacity_delta=x[10:11,:]
        return pos_delta,scales_delta,rotations_delta
        
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
    def __init__(self, out_ch=48):
        super(Decoder_Unet, self).__init__()
        # Decoder Layers
        self.decoder_layers = nn.ModuleList([
            UpsampleConvBlock(1024, 512),        # Layer 4
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
        self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

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


def load_image(path):
    image = np.clip(imread(path),0,None)
    # h,w,c = image.shape
    # image = np.resize(image,(w//2,h//2,3))
    if image.dtype == 'uint8':
        resized_image = torch.from_numpy(np.array(image)) / 255.0
        
    else:
        resized_image = torch.from_numpy(np.array(image)) 
    if len(resized_image.shape) == 3:
        
        return resized_image.permute(2, 0, 1).float().cuda()
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1).cuda()


class ModifiedEncoderUnet(Encoder_Unet):
    def __init__(self, pretrained_model_path, in_ch=9):
        # Initialize with arbitrary in_ch because we're going to override the first layer anyway
        super(ModifiedEncoderUnet, self).__init__(in_ch=1)

        # Load the state dictionary from the pre-trained model
        print("Path:",pretrained_model_path)
        state_dict = torch.load(pretrained_model_path, map_location='cpu')
        
        # Ensure 'enc_conv0.weight' exists and is a tensor
        if 'enc_conv0.weight' in state_dict and isinstance(state_dict['enc_conv0.weight'], torch.Tensor):
            original_enc_conv0_weight = state_dict['enc_conv0.weight']
            original_in_ch = original_enc_conv0_weight.size(1)

            # Proceed if adjustment is needed
            if in_ch != original_in_ch:
                # Adjusting the weight for new input channels
                # This repeats the first 'in_ch/original_in_ch' channels to fill up to 'in_ch'
                # For a more sophisticated approach, consider other methods of weight initialization
                new_channels = in_ch // original_in_ch
                extra_channels = in_ch % original_in_ch
                new_enc_conv0_weight = original_enc_conv0_weight.repeat(1, new_channels+1, 1, 1)[:, :in_ch, :, :]

                # Update the state_dict
                state_dict['enc_conv0.weight'] = new_enc_conv0_weight
        else:
            raise KeyError("enc_conv0.weight not found in the state dictionary")

        # Re-initialize the first layer correctly
        self.enc_conv0 = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=3, padding=1)

        # Load the updated state dict
        # Using strict=False allows for the architecture to be different
        self.load_state_dict(state_dict, strict=False)

def load_encoder(path,encoder):
        
    weight_dict = torch.load(path, map_location="cpu")
    print(weight_dict)
    encoder.load_state_dict(weight_dict)
    return encoder
    



if __name__ == "__main__":
    latent_size = 256  # Size of the latent vector
    lights = 1
    encoder = EnvMapEncoder()
    envmap = torch.rand((256,512,3)).permute((2,0,1))
    # latent_vector = encoder(envmap)
    latent_vector = torch.rand(1,1,256)
    view = torch.rand((1,1,3))
    
    gaussians = 256
    color = 3
    # color_decoder = Decoder_RGB(latent=latent_size,gaussians=gaussians)
    color_decoder = Decoder_RGB_SH(latent=latent_size,gaussians=gaussians,sh_degree=color)
    # opacity_decoder = Decoder_alpha(latent=latent_size,gaussians=gaussians)
    shape_decoder = Decoder_shape(latent=latent_size,gaussians=gaussians,shape_len=10)
    rgb = color_decoder(latent_vector)
    
    pos,scale,rot,opacity = shape_decoder(latent_vector)
    # decoder = Decoder_SH()
    # rgb = decoder(latent_vector)
    # print(rgb.shape)
    encoder_path = '/CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/full_light_sh_single_unet_exp/point_cloud/iteration_50000/encoder.pth'
    
    # encoder = Encoder_Unet(in_ch=6)
    # decoder = Decoder_Unet(out_ch=59)
    # encoder = load_encoder(encoder_path,encoder)
    # gauss_input = torch.rand((512,512,6)).permute((2,0,1))
    # encoder_feat,bottleneck = encoder(gauss_input.unsqueeze(0))
    # print(bottleneck.shape)
    
    # offset = decoder(encoder_feat,bottleneck)
    # print(offset.shape)
    # modified_model = ModifiedEncoderUnet(encoder_path, in_ch=9)
    # gauss_input = torch.rand((256,256,9)).permute((2,0,1))
    # encoder_feat,bottleneck = modified_model(gauss_input.unsqueeze(0))
    # print(bottleneck.shape)
    print(pos.shape)
    print(rgb.shape)
    print(count_parameters(encoder))
    print(count_parameters(shape_decoder))
    print(count_parameters(color_decoder))