import spectral 
import matplotlib.pyplot as plt
import numpy as np
img = spectral.envi.open('2023-10-12_007.hdr', '2023-10-12_007.raw')

# for i in range(0, img.shape[2]-1, 10):
#     p = img.read_band(i)
#     plt.imshow(p)
#     plt.show()
print(img.shape)
# print(np.array(img))
img1 = np.zeros([512, 512, 204])
print(img1)
for i in range(0, img.shape[2]):
    p = img.read_band(i)
    # print(p)
    img1[:,:,i] = p
    plt.imshow(p)
    plt.show()

print(img1)
print(img1.shape)
# print(img.read_band([]))