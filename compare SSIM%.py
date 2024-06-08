import matplotlib.pyplot as plt
import numpy as np

# Data from the table
categories = ['lighthouse', 'bridge', 'flamingo', '2024 image']
categories2 = ['lighthouse', 'bridge', 'flamingo', '2024\nimage']

rms_1layer = [8.65, 12.9, 11.8, 6.41]
ssim_1layer = [0.848, 0.764, 0.847, 0.922]

rms_2layer = [8.01, 12.56, 11.3, 6.09]
ssim_2layer = [0.868, 0.779, 0.859, 0.93]

rms_2layer_deblock = [7.97, 12.54, 11.29, 6.1]
ssim_2layer_deblock = [0.874, 0.78, 0.861, 0.931]
ssim_percentage_improvement = [0.7,0.1,0.2,0.1]

x = np.arange(len(categories))  # the label locations

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Plotting RMS error
ax1.plot(x, rms_1layer, 'o-', label='1-level DCT')
ax1.plot(x, rms_2layer, 'o-', label='2-level DCT')
ax1.plot(x, rms_2layer_deblock, 'o-', label='2-level DCT + deblock')

ax1.set_xlabel('Image')
ax1.set_ylabel('RMS error')
ax1.set_title('RMS error')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()

# Plotting SSIM
# ax2.plot(x, ssim_1layer, 'o-', label='1-level DCT')
# ax2.plot(x, ssim_2layer, 'o-', label='2-level DCT')


# Change to bar chart
ax2.bar(x, ssim_percentage_improvement, label='2-level DCT + deblock')

# ax2.set_xlabel('Image')
ax2.set_ylabel('SSIM % change')
ax2.set_ylim([0, 0.8])
ax2.set_title('SSIM % change after deblocking')
ax2.set_xticks(x)
ax2.set_xticklabels(categories2)
# ax2.legend()

fig.tight_layout()
plt.show()
