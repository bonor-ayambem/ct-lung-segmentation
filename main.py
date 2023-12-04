# !pip install SimpleITK -q
# !pip install scikit-image -q
# !pip install nilearn -q
# !pip install nibabel -q

import nilearn as nil
import nibabel as nib
from nilearn import plotting
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

datapath = "CTScans/lola11-01.mha"
img = sitk.ReadImage(datapath, sitk.sitkFloat32)

# The order of index and dimensions need careful attention during conversion!!
nda = sitk.GetArrayFromImage(img)
print("Size of ArrayFromImage:", nda.shape)
img = sitk.GetImageFromArray(nda)
print("Size of ImageFromArray:", img.GetSize())

affine = np.eye(4,4)
img_nii = nib.Nifti1Image(nda, affine)

plotting.plot_anat(img_nii, display_mode='x', cut_coords= 5,colorbar=False, output_file='image.png')

######################### thresholding_seg

# Load and get data
from nilearn.image import load_img

img_data = load_img(img_nii)
img_array = img_data.get_fdata()
# img_header = img_data.get_header()
print(img_array.shape)

# Get histogram and bin edges - numpy.histogram
hist, bins = np.histogram(img_array.ravel(), bins = 5)
print(hist)
print(bins)

# Computer the index of maximum frequency peak
max_hist_index = (np.argmax(hist))

# Get global threshold value
cut_index = max_hist_index+1
threshold_value_global = bins[cut_index]

# Plot histogram to find threshold value - plt.hist
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4), facecolor='white')

# Specify max number of bins
ax0.hist(img_array.ravel(), bins=5, histtype='stepfilled', facecolor='g', alpha=1)  # alpha as an integer between 0 and 1 specifying the transparency of each histogram.
ax0.set_title('pyplot histogram')

# Specify exact bins
ax1.hist(img_array.ravel(), bins=bins, histtype='stepfilled', facecolor='g', alpha=1)  # alpha as an integer between 0 and 1 specifying the transparency of each histogram.
ax1.set_title('numpy histogram')

# Create a histogram by providing the bin edges (unequally spaced).
spec_bins = [-3, 0, 6]
ax2.hist(img_array.ravel(), bins = spec_bins, histtype='bar', rwidth=0.8)
ax2.set_title('unequal bins')

fig.tight_layout()
print("about to plot")
plt.show()
print("after show")

# Global thresholding
from nilearn.image import load_img, math_img
from nilearn import plotting

print("here")
# manipulate with math_img, either data or filename

str_img = 'img > ' + str(threshold_value_global)
print(str_img)
mask = math_img(str_img, img=img_data)

# mask = math_img('img > 2.717', img=img_data)

str_img_abs = 'abs(img) > ' + str(threshold_value_global)
print(str_img_abs)
mask_abs = math_img(str_img_abs, img=tmap_filename)

## manipulate with math_img  - another example
# result_img = math_img("img1 - img2", img1=data1, img2=data1)

# Show global thresholding image in plot_stat_map mode
plotting.plot_stat_map(mask, display_mode='z', cut_coords=[10],
                       title='image masked with global threshold in plot_stat_map mode', colorbar=False, output_file='output1.png')

# Show global thresholding image in plot_roi mode (mask)
plotting.plot_roi(mask, display_mode='z', cut_coords=[10],
                       title='image masked with global threshold in plot_roi mode', colorbar=False, output_file='output2.png')

# Show absolute global thresholding image, with both positive and negative values exceeding threshold
plotting.plot_roi(mask_abs, display_mode='z', cut_coords=[10],
                       title='image masked with absolute global threshold in plot_roi mode', colorbar=False, output_file='output3.png')

# Show image with threshold value (same as absolute global thresholding)
plotting.plot_stat_map(img_data, display_mode='z', cut_coords=[10],
                        colorbar=True, threshold=threshold_value_global,
                        title='image masked with plotting threshold value', output_file='output4.png')

# Use bg_img to control background image, the default is <MNI152Template>
plotting.plot_stat_map(img_data, display_mode='z', cut_coords=[10], bg_img = None,
                        colorbar=False, threshold=threshold_value_global,
                        title='image masked with plotting threshold value BUT without background', output_file='output5.png')

# Global and Percentile thresholding

from nilearn.image import threshold_img

# Type 1: Global strategy used will be based on image intensity
threshold_value_img = threshold_img(tmap_filename, threshold=threshold_value_global, copy=False, output_file='output6.png') # absolute global threshold

# Type 2: Percentile strategy used will be based on score at percentile
threshold_percentile_img = threshold_img(tmap_filename, threshold='97%', copy=False, output_file='output7.png')

# Show global threshold image
plotting.plot_stat_map(threshold_value_img, display_mode='z', cut_coords=(10, 20, 30),
                       title='threshold image with absolute global', colorbar=True, vmax=9, output_file='output8.png')

# Show percentile threshold image
plotting.plot_stat_map(threshold_percentile_img, display_mode='z', cut_coords=(10, 20, 30),
                       title='threshold image with percentile', colorbar=True, vmax=9, output_file='output9.png')


# Adaptive/Local thresholding

# from skimage.filters import threshold_local  # Only support 2D image

# # img_2Darray = img_array.reshape(img_array.shape[0]*img_array.shape[2], img_array.shape[1])
# img_2Darray = img_array.transpose(2,0,1).reshape(-1, img_array.shape[1])

# binary_image = img_2Darray > threshold_local(img_2Darray, block_size = 99, method='mean', offset=-2)
# bool_val = binary_image.astype(int)

# # Reshapce back to 3D image
# img_3Darray = bool_val.reshape(np.roll(img_array.shape,1)).transpose(1,2,0)

# ## Transform data into nifty format (Important!!!)
# nifty_img = nib.Nifti1Image(img_3Darray, img_data.affine)

# # Show global thresholding image
# plotting.plot_roi(mask, display_mode='z', cut_coords=(10, 20, 30),
#                        title='threshold image with global cut', colorbar=False)

# # Show adaptive thresholding image
# plotting.plot_roi(nifty_img, display_mode='z', cut_coords=(10, 20, 30),
#                        title='threshold image with local cut', colorbar=True, vmax=9)