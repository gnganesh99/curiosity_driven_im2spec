import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


#from im2spec.im2spec.models import conv_block, dilated_block

import atomai
from atomai.nets.blocks import ResBlock, ResModule
from atomai.nets.blocks import ConvBlock as conv_block
from atomai.nets.blocks import DilatedBlock as dilated_block


class im2spec(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size,
                 target_size: int,
                 latent_dim: int,
                 nb_filters_enc: int = 128,
                 nb_filters_dec: int = 64) -> None:
        
        super(im2spec, self).__init__()
        
        self.n, self.m = feature_size
        
        self.ts = target_size
        
        self.e_filt = nb_filters_enc
        
        self.d_filt = nb_filters_dec
        # Encoder params
        
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, batch_norm=True, dropout_ = 0.5)
        
        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)
        # Decoder params
        
        
        #Wrap the encoder function into an `nn.Module`
        self.encoder = nn.Module()
        self.encoder.forward = lambda x: self._encoder(x)
        
        self.dec_fc1 = nn.Linear(latent_dim, self.ts //4 )
        self.dec_fc2 = nn.Linear(self.ts // 4, self.ts //4 * 2 )
        self.dec_fc3 = nn.Linear(self.ts //4 * 2, self.ts //4 * 3 )
        self.dec_fc4 = nn.Linear(self.ts //4 * 3, self.ts)
        self.dec_fc5 = nn.Linear(self.ts, self.ts)
        self.dec_fc6 = nn.Linear(self.ts, self.ts)
        
       
    def _encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        x = self.enc_conv(features)
        x = x.reshape(-1, self.e_filt * self.m * self.n)
        return self.enc_fc(x)

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """
        
        x = F.relu(self.dec_fc1(encoded))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = F.relu(self.dec_fc4(x))
        x = F.relu(self.dec_fc5(x))
        
        return self.dec_fc6(x).reshape(-1, self.ts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""

        with torch.no_grad():
            return self.forward(x)
        



class im2spec_2(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size,
                 target_size: int,
                 latent_dim: int,
                 nb_filters_enc: int = 128) -> None:
        
        super(im2spec_2, self).__init__()
        
        self.n, self.m = feature_size
        
        self.ts = target_size
        
        self.e_filt = nb_filters_enc
        
        # Encoder params
        
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.2, batch_norm=True, dropout_ = 0.1)
        
        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)
        
        #Wrap the encoder function into an `nn.Module`
        self.encoder = nn.Module()
        self.encoder.forward = lambda x: self._encoder(x)
        
        
        
        # Decoder params
        
        self.dec_fc1 = nn.Linear(latent_dim, self.ts //4 )
        self.dec_fc2 = nn.Linear(self.ts // 4, self.ts //4 * 2 )
        self.dec_fc3 = nn.Linear(self.ts //4 * 2, self.ts //4 * 3 )
        self.dec_fc4 = nn.Linear(self.ts //4 * 3, self.ts)
        self.dec_fc5 = nn.Linear(self.ts, self.ts)
        
       
    def _encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        x = self.enc_conv(features)
        x = x.reshape(-1, self.e_filt * self.m * self.n)
        return self.enc_fc(x)

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """
        
        x = F.relu(self.dec_fc1(encoded))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = F.relu(self.dec_fc4(x))
        
        
        return self.dec_fc5(x).reshape(-1, self.ts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""

        with torch.no_grad():
            return self.forward(x)
        


class im2spec_3(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size,
                 target_size: int,
                 latent_dim: int,
                 nb_filters_enc: int = 128) -> None:
        
        super(im2spec_3, self).__init__()
        
        self.n, self.m = feature_size
        
        self.ts = target_size
        
        self.e_filt = nb_filters_enc
        
        # Encoder params
               
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.2, batch_norm=True, dropout_ = 0.1)
        

        self.enc_atrous = dilated_block(
            ndim=2, input_channels=self.e_filt, output_channels=self.e_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, batch_norm=True)


        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)
        
        
        #Wrap the encoder function into an `nn.Module`
        self.encoder = nn.Module()
        self.encoder.forward = lambda x: self._encoder(x)
        
        
        
        
        
        # Decoder params        
        self.dec_fc1 = nn.Linear(latent_dim, self.ts //4 )
        self.dec_fc2 = nn.Linear(self.ts // 4, self.ts //4 * 2 )
        self.dec_fc3 = nn.Linear(self.ts //4 * 2, self.ts //4 * 3 )
        self.dec_fc4 = nn.Linear(self.ts //4 * 3, self.ts)
        self.dec_fc5 = nn.Linear(self.ts, self.ts)
        
        

        
       
    def _encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """

        x = self.enc_conv(features)
        x = self.enc_atrous(x)
        x = x.reshape(-1, self.e_filt * self.m * self.n) # this retains the batch size
        
        
        return self.enc_fc(x)

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """
        x = F.relu(self.dec_fc1(encoded))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = F.relu(self.dec_fc4(x))
        x = self.dec_fc5(x)
        
#         x = F.relu(self.dec_fc1(encoded))
#         x = F.relu(self.dec_fc2(x))
#         x = F.relu(self.dec_fc3(x))
        
        
        
        return x.reshape(-1, self.ts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""

        with torch.no_grad():
            return self.forward(x)
        


class im2spec_4(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size,
                 target_size: int,
                 latent_dim: int,
                 nb_filters_enc: int = 128) -> None:
        
        super(im2spec_4, self).__init__()
        
        self.n, self.m = feature_size
        
        self.ts = target_size
        
        self.e_filt = nb_filters_enc
        
        # Encoder params
        

        self.enc_Resmod = ResModule(
            ndim=2, res_depth=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, batch_norm=True)
        
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=self.e_filt, output_channels=self.e_filt,
            lrelu_a=0.2, batch_norm=True, dropout_ = 0.2)

        self.enc_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_fc = nn.Linear(self.e_filt * self.n//2 * self.m//2, latent_dim)
        
        
        #Wrap the encoder function into an `nn.Module`
        self.encoder = nn.Module()
        self.encoder.forward = lambda x: self._encoder(x)
        
        
        
        
        # Decoder params        
        self.dec_fc1 = nn.Linear(latent_dim, self.ts //4 )
        self.dec_fc2 = nn.Linear(self.ts // 4, self.ts //4 * 2 )
        self.dec_fc3 = nn.Linear(self.ts //4 * 2, self.ts //4 * 3 )
        self.dec_fc4 = nn.Linear(self.ts //4 * 3, self.ts)
        self.dec_fc5 = nn.Linear(self.ts, self.ts)
        
        
       
    def _encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        x = self.enc_Resmod(features)
        x = self.enc_conv(x)
        x = self.enc_pool(x)

        N = x.size(0)
        x = x.view(N, -1)

        return F.relu(self.enc_fc(x))

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """
        #introduce non-linearity
        x = F.leaky_relu(self.dec_fc1(encoded), negative_slope=0.1)
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = F.relu(self.dec_fc4(x))
        x = self.dec_fc5(x)
        
        return x.reshape(-1, self.ts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""

        with torch.no_grad():
            return self.forward(x)
        

class im2spec_5(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size,
                 target_size: int,
                 latent_dim: int,
                 nb_filters_enc: int = 64) -> None:
        
        super(im2spec_5, self).__init__()
        
        self.n, self.m = feature_size
        
        self.ts = target_size
        
        self.e_filt = nb_filters_enc
        
        # Encoder params
        

        self.enc_Resmod = ResModule(
            ndim=2, res_depth=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, batch_norm=True)

        self.enc_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc_atrous = dilated_block(
            ndim=2, input_channels=self.e_filt, output_channels=self.e_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, batch_norm=True)
        
        self.enc_fc = nn.Linear(self.e_filt * self.n//2 * self.m//2, latent_dim)
        
        
        
        #Wrap the encoder function into an `nn.Module`
        self.encoder = nn.Module()
        self.encoder.forward = lambda x: self._encoder(x)
        
        
        
        # Decoder params
        self.dec_fc1 = nn.Linear(latent_dim, self.ts //4 )
        self.dec_fc2 = nn.Linear(self.ts // 4, self.ts //4 * 2 )
        self.dec_fc3 = nn.Linear(self.ts //4 * 2, self.ts)
        self.dec_fc4 = nn.Linear(self.ts, self.ts)        
        
       
    def _encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        x = self.enc_Resmod(features)
        x = self.enc_pool(x)
        x= self.enc_atrous(x)

        N = x.size(0)
        x = x.view(N, -1)

        return self.enc_fc(x)

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """
        
        x = F.relu(self.dec_fc1(encoded))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = F.relu(self.dec_fc4(x))
        
        
        return x.reshape(-1, self.ts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""

        with torch.no_grad():
            return self.forward(x)
        


class ensemble_im2spec(nn.Module):

    def __init__(self, input_dim, out_put_dim, models = (im2spec_3, im2spec_4), force_latent_dim = None):
        super(ensemble_im2spec, self).__init__()

        if force_latent_dim is not None:
            self.models = [model(input_dim, out_put_dim, latent_dim = int(force_latent_dim)) for model in models]
            
        else:
            self.models = [model(input_dim, out_put_dim, latent_dim = np.random.randint(2, 8)) for model in models]

    def forward(self, x):
        pred = [model(x) for model in self.models]
        return pred

    def train(self):
        [model.train() for model in self.models]
    
    def eval(self):
        [model.eval() for model in self.models]

    def predict(self, x):
        
        [model.eval() for model in self.models]

        with torch.no_grad():
        
            return self.forward(x)
    
    def to(self, device):
        
        [model.to(device) for model in self.models]
        

# class error_model(nn.Module):

#     def __init__(self, in_dim):
#         super(error_model, self).__init__()
#         self.encoder = im2spec(in_dim, target_size=1, latent_dim = 64).encoder
#         self.fc1 = nn.Linear(64, 32)
#         self.fc2 = nn.Linear(32, 1)
        
#     def forward(self, x):

#         x = x.unsqueeze(1)
#         x = self.encoder(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))

#         return x.reshape(-1)
    
#     def to(self, device):
#         for param in self.parameters():
#             param.data = param.data.to(device)
#             if param.grad is not None:
#                 param.grad.data = param.grad.data.to(device)
        
class error_model(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size,
                 target_size: int = 1,
                 latent_dim: int = 32,
                 nb_filters_enc: int = 128,
                 nb_filters_dec: int = 64) -> None:
        
        super().__init__()
        
        self.n, self.m = feature_size
        
        self.ts = target_size
        
        self.e_filt = nb_filters_enc
        
        self.d_filt = nb_filters_dec
        # Encoder params
        
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, batch_norm=True, dropout_ = 0.1)
        
        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)
        # Decoder params
        
        self.dec_fc1 = nn.Linear(latent_dim, 32 )
        self.dec_fc2 = nn.Linear(32, 16 )
        self.dec_fc3 = nn.Linear(16, 8)
        self.dec_fc4 = nn.Linear(8, 1) 
        
 
        
       
    def encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        x = self.enc_conv(features)
        x = x.reshape(-1, self.e_filt * self.m * self.n)
        return self.enc_fc(x)

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """

        x = F.relu(self.dec_fc1(encoded))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = F.relu(self.dec_fc4(x))  
        
        
        return x.reshape(-1, self.ts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""

        with torch.no_grad():
            return self.forward(x)

        
        

class ensemble_error_model(nn.Module):

    def __init__(self, in_dim, n_models, model = error_model):
        super().__init__()

        self.models = [model(in_dim) for i in range(n_models)]
        
    def forward(self, x):

        ensemble_error = [model(x) for model in self.models]

        return ensemble_error
    
    def predict(self, x):

        [model.eval() for model in self.models]

        with torch.no_grad():
            return self.forward(x)
        
    def train(self):
        
        [model.train() for model in self.models]
    
    def eval(self):

        [model.eval() for model in self.models]

    def to(self, device):
        
        [model.to(device) for model in self.models]
        
        
        


class DecoderModule(nn.Module):
    def __init__(self, embed_dim, target_size=1):
        
        super(DecoderModule, self).__init__()
        
        self.ts = target_size
        
        self.dec_fc1 = nn.Linear(embed_dim, 32)
        self.dec_fc2 = nn.Linear(32, 16)
        self.dec_fc3 = nn.Linear(16, 8)
        self.dec_fc4 = nn.Linear(8, target_size)

    def forward(self, embedding):
        
        x = F.relu(self.dec_fc1(embedding))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = self.dec_fc4(x)
        
        return x.reshape(-1, self.ts)
            
            
        
class CustomDecoder(nn.Module):
    def __init__(self, encoder, embed_dim, target_size=1):
        super(CustomDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = DecoderModule(embed_dim, target_size)  # Now a module

    def forward(self, x):
        x = x.unsqueeze(1)
        embedding = self.encoder(x)
        return self.decoder(embedding)

    def train_only_decoder(self):
        
        # Freeze encoder params
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder.train()
        for param in self.decoder.parameters():
            param.requires_grad = True
            
    def predict(self, x: torch.Tensor):
        """Predict spectra from image"""

        with torch.no_grad():
            return self.forward(x)


