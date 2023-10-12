import torch
import torchvision
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
from stereo_x import StereoX
from process_data import process_stereo_data


model = StereoX(50)
model.load_state_dict(torch.load('StereoX.pt'))
model.eval()

rs = torchvision.transforms.Resize((37, 122))
train_loader, val_loader, norm_constants = process_stereo_data("../data_scene_flow/training/")
print("NORM CONST: ", norm_constants)
std = norm_constants['std'].item()
mean = norm_constants['mean'].item()

"""cnt = 5
for batch_i, data in enumerate(train_loader):
    C, L, R = model(data['L_img'])
    #Cr = (C[0].detach().numpy() * std) + mean
    #Lr = (L[0].detach().numpy() * std) + mean
    #Rr = (R[0].detach().numpy() * std) + mean
    Cr = (C[0][0].permute(1, 2, 0).detach().numpy() * std) + mean
    Lr = (L[0][0].permute(1, 2, 0).detach().numpy() * std) + mean
    Rr = (R[0][0].permute(1, 2, 0).detach().numpy() * std) + mean
    print(Cr.shape)
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.imshow(Cr/np.max(Cr))
    ax2.imshow(Lr/np.max(Lr))
    ax3.imshow(Rr/np.max(Rr))
    plt.show()
    cnt += 1
    if cnt == 6:
        break"""

start = 10
fig, ax = plt.subplots(5, 4)
fig.suptitle("Qualitative Performance on Mixed Training/Validation Data", fontsize=16)
ax[0][0].set_title("Actual Image")
ax[0][1].set_title("Predicted Left Image")
ax[0][2].set_title("Reconstructed Image")
ax[0][3].set_title("Predicted Right Image")
for item in range(start, start+5):
    s = str(item).zfill(6) + '_1' + str(item%2) + '.png'
    path_L = "../data_scene_flow/training/" + "image_L/" + s
    img_L = rs(read_image(path_L)).float()
    # Each VAE returns: reconstructed_state, mu, log_var, latent_state
    C, L, R = model(img_L)
    Cr = (C[0].permute(1, 2, 0).detach().numpy() * std) + mean
    Lr = (L[0].permute(1, 2, 0).detach().numpy() * std) + mean
    Rr = (R[0].permute(1, 2, 0).detach().numpy() * std) + mean
    print(img_L)
    ax[item-start][0].imshow(img_L.permute(1, 2, 0).int().detach().numpy())
    ax[item-start][2].imshow(Cr/np.max(Cr))
    ax[item-start][1].imshow(Lr/np.max(Lr))
    ax[item-start][3].imshow(Rr/np.max(Rr))
plt.show()