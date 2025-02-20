# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:45:26 2024

@author: ismt
"""
from utils2 import make_env

import matplotlib.pyplot as plt

# Get observation from environment
env = make_env('PongNoFrameskip-v4')
obs, _ = env.reset()

# Take a few random actions to get some movement
for _ in range(20):
    obs, _, _, _, _ = env.step(1)

# Plot all 4 frames side by side
plt.figure(figsize=(15, 4))

for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(obs[:,:,i], cmap='gray')  # Get i'th channel
    plt.title(f'Frame {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Plot just one frame in detail
plt.figure(figsize=(8, 8))
plt.imshow(obs[:,:,0], cmap='gray')  # Show first frame
plt.title('Detailed view of Frame 1')
plt.colorbar()  # Shows the scale of pixel values (0-1)
plt.axis('off')
plt.show()

# Print shape information
print("Full observation shape:", obs.shape)
print("Single frame shape:", obs[:,:,0].shape)
print("Value range:", obs.min(), "-", obs.max())