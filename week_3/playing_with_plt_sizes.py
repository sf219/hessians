import matplotlib.pyplot as plt
import numpy as np

top=0.673
bottom=0.061
left=0.009
right=0.983
hspace=0.2
wspace=0.0

plt.figure(figsize=(16.5, 3))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)

fig_plots = np.random.randn(6, 1024, 1024)

plt.subplot(1, 6, 1)
im = plt.imshow(fig_plots[0])
plt.title('Image', fontsize=16)
plt.xticks([])
plt.yticks([])

plt.subplot(1, 6, 2)
plt.imshow(fig_plots[1], cmap='gray')
plt.title('SSIM', fontsize=16)
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(1, 6, 3)
plt.imshow(fig_plots[2], cmap='gray')
plt.title('MS-SSIM', fontsize=16)
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(1, 6, 5)
plt.imshow(fig_plots[3], cmap='gray')
plt.title('BRISQUE', fontsize=16)
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(1, 6, 4)
plt.imshow(fig_plots[4], cmap='gray')
plt.title('LPIPS', fontsize=16)
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(1, 6, 6)
plt.imshow(fig_plots[5], cmap='gray')
plt.title('NIQE', fontsize=16)
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.suptitle('Diagonal estimator', fontsize=18)

plt.tight_layout()

plt.show()