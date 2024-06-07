import matplotlib.pyplot as plt
import numpy as np

# Data from the table
categories = ['lighthouse', 'bridge', 'flamingo', '2024 image']

rms_2layer = [8.01, 12.56, 11.3, 6.09]
ssim_2layer = [0.868, 0.779, 0.859, 0.93]

rms_2layer_deblock = [7.97, 12.54, 11.29, 6.1]
ssim_2layer_deblock = [0.874, 0.78, 0.861, 0.931]

x = np.arange(len(categories))  # the label locations

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Plotting RMS error
ax1.plot(x, rms_2layer, 'o-', label='2-level DCT')
ax1.plot(x, rms_2layer_deblock, 's-', label='2-level DCT + deblock')

ax1.set_xlabel('Image')
ax1.set_ylabel('RMS error')
ax1.set_title('RMS error by category and method')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()

# Plotting SSIM
ax2.plot(x, ssim_2layer, 'o-', label='2-level DCT')
ax2.plot(x, ssim_2layer_deblock, 's-', label='2-level DCT + deblock')

ax2.set_xlabel('Categories')
ax2.set_ylabel('Image')
ax2.set_title('SSIM by category and method')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()

fig.tight_layout()
plt.show()
