import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from PIL import Image
from numpy import asarray

im1 = Image.open('./plotss/cnn_softmax_accuracy.png')
im2 = Image.open('./plotss/intensity_softmax_accuracy.png')
im3 = Image.open('./plotss/intensity_svm_accuracy.png')
im4 = Image.open('./plotss/rgb_softmax_accuracy.png')
im5 = Image.open('./plotss/rgb_svm_accuracy.png')
fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5, sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(im1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.imshow(im2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.imshow(im3, cmap=plt.cm.gray)
ax4.axis('off')
ax4.imshow(im4, cmap=plt.cm.gray)
ax5.axis('off')
ax5.imshow(im5, cmap=plt.cm.gray)
plt.show()

# image1 = asarray(Image.open('../CURE-TSR/Real_Train/ChallengeFree/01_06_00_00_0001.bmp'))
# image2= asarray(Image.open('../CURE-TSR/Real_Train/CodecError-1/01_06_03_01_0001.bmp'))
# image3= asarray(Image.open('../CURE-TSR/Real_Train/CodecError-2/01_06_03_02_0001.bmp'))
# image4= asarray(Image.open('../CURE-TSR/Real_Train/ChallengeFree/01_02_00_00_0001.bmp'))
#
# fd1, hog_image1 = hog(image1, orientations=8, pixels_per_cell=(6, 6),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)
#
# fd2, hog_image2 = hog(image2, orientations=8, pixels_per_cell=(6, 6),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)
# fd3, hog_image3 = hog(image3, orientations=8, pixels_per_cell=(6, 6),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)
#
# fd4, hog_image4 = hog(image4, orientations=8, pixels_per_cell=(6, 6),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)
#
# fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, sharex=True, sharey=True)
#
# # ax1.axis('off')
# # ax1.imshow(image1, cmap=plt.cm.gray)
# # ax1.set_title('Input image')
#
# # Rescale histogram for better display
# hog_image_rescaled1 = exposure.rescale_intensity(hog_image1, in_range=(0, 10))
# hog_image_rescaled2= exposure.rescale_intensity(hog_image2, in_range=(0, 10))
# hog_image_rescaled3 = exposure.rescale_intensity(hog_image3, in_range=(0, 10))
# hog_image_rescaled4 = exposure.rescale_intensity(hog_image4, in_range=(0, 10))
#
# ax1.axis('off')
# ax1.imshow(hog_image_rescaled1, cmap=plt.cm.gray)
# ax1.set_title('Codec HoG-No Challenge')
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled2, cmap=plt.cm.gray)
# ax2.set_title('Codec HoG-Challenge Level1')
# ax3.axis('off')
# ax3.imshow(hog_image_rescaled3, cmap=plt.cm.gray)
# ax3.set_title('Codec HoG-Challenge Level2')
#
# ax4.axis('off')
# ax4.imshow(hog_image_rescaled4, cmap=plt.cm.gray)
# ax4.set_title('Speed Limit HoG-No Challenge')
# plt.show()