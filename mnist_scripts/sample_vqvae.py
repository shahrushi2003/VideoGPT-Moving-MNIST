# %%
import matplotlib.pyplot as plt

from videogpt import VQVAE, VideoData

# %%

CHECKPOINT_PATH = ...
DEVICE = ...

# %%
model = VQVAE.load_from_checkpoint(CHECKPOINT_PATH)

# disable randomness, dropout, etc...
model.eval()

# %%
data = VideoData(model.args)

val_dataset = data._dataset(False)

# %%
# Should be: (10000, torch.Size([3, 16, 64, 64]))
print(len(val_dataset), val_dataset[0]['video'].shape)

index = 0

fig, ax = plt.subplots(1, 16, figsize=(16, 1))
original = val_dataset[index]['video'].permute(1, 2, 3, 0).numpy()

for i in range(16):
    ax[i].imshow(original[i])
    ax[i].axis("off")

# %%
device = DEVICE

model = model.to(device)

# %%
# Feed data into model and get the x_recon and plot it (second output from forward)

x = val_dataset[index]['video'].unsqueeze(0)
x = x.to(device)

_, x_recon, _ = model(x)

# %%
fig, ax = plt.subplots(1, 16, figsize=(16, 1))
video = x_recon[0].permute(1, 2, 3, 0).cpu().detach().numpy()

for i in range(16):
    ax[i].imshow(video[i])
    ax[i].axis("off")

plt.show()

# %%
# Make a figure that plots both original and reconstructed video

fig, ax = plt.subplots(2, 16, figsize=(16, 2))

for i in range(16):
    ax[0, i].imshow(original[i])
    ax[0, i].axis("off")
    ax[1, i].imshow(video[i])
    ax[1, i].axis("off")

plt.show()
