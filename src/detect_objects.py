import numpy as np
from skimage.util import invert
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage import binary_fill_holes
from skimage.morphology import erosion,disk
import matplotlib.pyplot as plt

def detect_objects_on_rgb_img(rgb_img):
    gray_img = rgb2gray(rgb_img)

    thresh = threshold_otsu(gray_img)
    # imgs are black on white
    binary = gray_img < thresh
    # eroded = erosion(binary, disk(5))
    filled = binary_fill_holes(binary)
    labeled_image = label(binary)
    return labeled_image

def detect_objects_on_rgb_img_with_watershed(mask):
    import scipy.ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from skimage.util import img_as_uint

    mask =img_as_uint(mask)

    plt.imshow(mask)
    plt.show()
    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    print("reached")
    labels = watershed(-distance, markers, mask=mask)
    return labels

def detect_objects_on_rgb_img_with_cellpose(img, diameter=100):
    from cellpose import models

    inverted_img =  1 - img

    model = models.Cellpose(gpu=False, model_type="nuclei")
    channels = [0,0]
    mask, flows, styles, diams = model.eval(inverted_img, diameter=diameter, channels=channels)

    return mask

def detect_objects_with_dbscan(img, sigma, pixel_density, block_size = 51, debug=False):
    from sklearn.cluster import DBSCAN
    from scipy.ndimage import center_of_mass, label
    from skimage.filters import gaussian, threshold_otsu
    from skimage import feature

    def subsample(img):
        # c
        x=img.shape[0]
        y=img.shape[1]
        crop_x=int(x/3)
        crop_y=int(y/3)
        start_x=int((x/2)-(crop_x/2))
        start_y=int((y/2)-(crop_y/2))
        return img[start_x:start_x+crop_x, start_y:start_y+crop_y]

    inverted_img =  1 - img

    # If the image is already single channel, this will throw an error
    try:
        grayscale = rgb2gray(inverted_img)
    except ValueError:
        grayscale = inverted_img

    gauss = gaussian(grayscale, sigma=sigma)
    img_subsampled = subsample(gauss)

    if debug:
        fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
        axs = axs.flatten()
        axs[0].imshow(img)
        axs[0].set_title("original")
        axs[1].imshow(grayscale)
        axs[1].set_title("gray")


        axs[2].imshow(gauss)
        axs[2].set_title("gauss")

        axs[3].imshow(img_subsampled)
        axs[3].set_title("subsampled")

    # thresh =threshold_otsu(img_subsampled)
    thresh = threshold_local(gauss, block_size)

    # Used to be without the tmp and masking, but skimage has removed the indices 
    tmp_is_peak = feature.peak_local_max(gauss, min_distance = int(2.5 * pixel_density), threshold_abs = thresh,#threshold_abs=thresh + (10*thresh)/100,
                                      exclude_border=False)
    is_peak = np.zeros_like(gauss, dtype=bool)
    is_peak[tuple(tmp_is_peak.T)] = True

    plabels = label(is_peak)[0]
    merged_peaks = center_of_mass(is_peak, plabels, range(1, np.max(plabels)+1))
    local_maxi = np.array(merged_peaks)

    if debug:
        axs[4].imshow(gauss)
        axs[4].scatter(local_maxi[:,1], local_maxi[:,0], color="r", s=1)
        axs[4].set_title("Spots before segmenting")

    X = local_maxi.copy()

    if len(local_maxi) > 0:
        # Compute DBSCAN
        db = DBSCAN(eps=20.6*pixel_density, min_samples=2).fit(local_maxi)
        labels = db.labels_
        # local maxi that aren't segmented (standalone maxi) are -1, so we don't wanna take the mean of that object
        # alse, labels starts counting at 0, which can be annoying downstream, so we add 1 to all 

        # We want the -1's to be seperate objects, so we relabel the local_maxi 
        local_maxi = np.hstack((local_maxi, np.expand_dims(labels,1)))


        # Get the unique values in the third column, excluding -1
        # This also fixes the 0's cause we start counting at 1
        unique_labels = np.unique(local_maxi[local_maxi[:, 2] != -1, 2])

        # Create a dictionary mapping for the unique labels that aren't -1
        label_mapping = {label: i for i, label in enumerate(unique_labels, 1)}

        # Counter for unique labels for -1
        new_label_counter = max(label_mapping.values()) + 1

        # Iterate over the array and assign new labels for each -1
        for i in range(len(local_maxi)):
            if local_maxi[i, 2] == -1:
                local_maxi[i, 2] = new_label_counter
                new_label_counter += 1
            else:
                # Replace existing label with mapped label if not -1
                local_maxi[i, 2] = label_mapping[local_maxi[i, 2]]

        # Only keep middlepoints of segmented objects

        unique_values = np.unique(local_maxi[:, 2])

        # Dictionary to store midpoints
        midpoints = {}

        for value in unique_values:
            # Filter rows with the current unique value
            rows = local_maxi[local_maxi[:, 2] == value]
            # Calculate the midpoint by averaging the first two columns
            midpoint = rows[:, :2].mean(axis=0)
            midpoints[value] = midpoint


        combined_local_maxi = np.array([[x, y, label] for label, (x, y) in midpoints.items()])

        if debug:
            axs[5].imshow(img)
            axs[5].scatter(combined_local_maxi[:,1], combined_local_maxi[:,0], color="r", s=1)
            axs[5].set_title("Spots after merging close dots")
            plt.title("object detection with DBscan")
            plt.show()

        return combined_local_maxi




if __name__ == '__main__':
    from skimage import io
    from utils import convertCziToUsableArray
    from aicspylibczi import CziFile
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    from skimage.util import img_as_uint
    import glob

    img_path = "../test_photos/LF6_05_ah_3.czi"
    czi = CziFile(str(img_path))
    bad_img = convertCziToUsableArray(czi)

    img_path = "../test_photos/UL13783_11_ah_3.czi"
    czi = CziFile(str(img_path))
    good_img = convertCziToUsableArray(czi)

    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(bad_img)
    # axs[1].imshow(good_img)
    # plt.show()

    detect_objects_with_dbscan(bad_img, sigma=10, pixel_density=2, debug=True)

