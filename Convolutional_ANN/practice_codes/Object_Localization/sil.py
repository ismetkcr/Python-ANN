import numpy as np
import matplotlib.pyplot as plt

# Create a simple background image (8x8 pixels, RGB)
background = np.zeros((8, 8, 3), dtype=np.uint8)
background[2:6, 2:6] = [255, 255, 0]  # Yellow square in the middle

# Create an object image with transparency (4x4 pixels, RGBA)
# The object will be a red square with a transparent border
obj = np.zeros((4, 4, 4), dtype=np.uint8)
obj[1:3, 1:3] = [255, 0, 0, 255]  # Red square in the center, alpha = 255 (opaque)
obj[:,:,3] = np.array([
    [0, 0, 0, 0],
    [0, 255, 255, 0],
    [0, 255, 255, 0],
    [0, 0, 0, 0]
], dtype=np.uint8)  # Alpha channel

# Display the background and the object
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.title('Background')
plt.imshow(background)
plt.subplot(1, 2, 2)
plt.title('Object (RGBA)')
plt.imshow(obj)
plt.show()

# Calculate the mask and place the object on the background
mask = (obj[:, :, 3] == 0)  # Transparent where alpha == 0
bg_slice = background[2:6, 2:6, :].copy()
bg_slice = np.expand_dims(mask, -1) * bg_slice  # Apply mask to background slice
bg_slice += obj[:, :, :3]  # Add the RGB channels of the object
background[2:6, 2:6, :] = bg_slice  # Place the slice back into the background

# Display the result
plt.title('Result')
plt.imshow(background)
plt.show()
