import torch
import torch.nn as nn
from vae import StateVAE


class StereoX(nn.Module):

    def __init__(self):
        self.monoc = StateVAE(15)
        self.LR = StateVAE(15)
        #self.RR = StateVAE(15)
        self.RL = StateVAE(15)

    def forward(self, L_img, R_img=None):
        # Each VAE returns: reconstructed_state, mu, log_var, latent_state
        if R_img is not None:
            LL = self.monoc(L_img)
            LR = self.LR(R_img)
            RR = self.monoc(R_img)
            RL = self.RL(L_img)
            return LL, LR, RR, RL
        LL = self.monoc(L_img)
        L = self.LR(L_img)
        R = self.RL(L_img)
        return LL, L, R
