import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from process_data import NormalizationTransform, process_stereo_data
from stereo_x import StereoX
from loss_functions import VAELoss

# Train the VAE

LR = 0.001
NUM_EPOCHS = 2000
BETA = 0.001
LATENT_DIM = 10
NUM_CHANNELS = 1
NUM_STEPS = 1
LATENT_DIM = 16

train_loader, val_loader, norm_constants = process_stereo_data("../data_scene_flow/training/")
norm_tr = NormalizationTransform(norm_constants)

#all_states = []
#for batch_i in train_loader:
#    state_i = batch_i['states']
#    all_states.append(state_i)
#all_states = torch.cat(all_states, axis=0)

#states = all_states.flatten(end_dim=1) # Get initial states (Num satates, num_channels, w, h)


#vae_model = StateVAE(latent_dim=LATENT_DIM, num_channels=1)
model = StereoX(50)
loss_func = VAELoss(beta=BETA)
optimizer = optim.Adam(model.parameters(), lr=LR)
pbar = tqdm(range(NUM_EPOCHS))
train_losses = []
for epoch_i in pbar:
    train_loss_i = 0.
    # --- Your code here
    cnt = 0
    for batch_i, data in enumerate(train_loader):
      optimizer.zero_grad()

      #states = data['states']
      #targets = data['actions']
      L_img = data['L_img']
      R_img = data['R_img']

      #reconstructed_states, mu, log_var, latent_state = vae_model(states)
      LL, LR, RR, RL = model(L_img, R_img)

      loss = loss_func(LL[0], L_img, LL[1], LL[2]) + loss_func(LR[0], L_img, LR[1], LR[2]) + loss_func(RR[0], R_img, RR[1], RR[2]) + loss_func(RL[0], R_img, RL[1], RL[2])

      loss.backward()
      optimizer.step()

      train_loss_i += loss.item()
      cnt += 1

    train_loss_i /= cnt


    # ---
    train_loss_i += loss.item()
    pbar.set_description(f'Latent dim {LATENT_DIM} - Loss: {train_loss_i:.4f}')
    train_losses.append(train_loss_i)

losses = train_losses
#vaes = vae_model
# Evaluate:
#vae_model.eval()
#states_rec, mu, log_var, latent_state = vae_model(states)


# plot train loss and test loss:
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
axes = [axes]
axes[0].plot(losses, label=f'latent_dim: {LATENT_DIM}')
axes[0].grid()
axes[0].legend()
axes[0].set_title('Train Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Train Loss')
axes[0].set_yscale('log')
plt.show()


# ---

# save model:
torch.save(model.state_dict(), 'StereoX.pt')