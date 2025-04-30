import pandas as pd
import numpy as np
from glob import glob
from skimage import io
from pathlib import Path
from load_image import loadImage
from process_image import normImage, convert_czi_to_img
import matplotlib.pyplot as plt
from aicspylibczi import CziFile
from skimage.color import rgb2hed, hed2rgb
from training_data_utils import measureLabeledImage, detect_areas_on_rgb_img, cutBboxesFromCenter
from tqdm import tqdm
from pathlib import Path
from skimage.util import img_as_ubyte


def extract_hematoxylin(img):
    # Separate the stains from the IHC image
    img_hd = rgb2hed(img)

    null = np.zeros_like(img_hd[:, :, 0])

    img_h = hed2rgb(np.stack((img_hd[:, :, 0], null, null), axis=-1))
    # img_e = hed2rgb(np.stack((null, img_hd[:, :, 1], null), axis=-1))
    # img_d = hed2rgb(np.stack((null, null, img_hd[:, :, 2]), axis=-1))
    return img_h


n_keep = 125
imgs = sorted(glob("/mnt/data/tool/geethika_project/vessels_nonneuron_neuron_HE/detection_results/*.czi"))

for img_path in tqdm(imgs):
    corresponding_labels_path = Path(img_path.replace("czi", "czi Detections.txt"))
    czi = CziFile(img_path)
    img_path = Path(img_path)
    out_dir =Path( ("./vessels_single_objects/"))
    out_dir_objects = out_dir / "objects"
    out_dir_labeled =Path( out_dir / "labeled_images")
    out_dir_measured =Path( out_dir / "measured_dfs")
    out_dir.mkdir(exist_ok=True)
    out_dir_labeled.mkdir(exist_ok=True)
    out_dir_measured.mkdir(exist_ok=True)
    out_dir_objects.mkdir(exist_ok=True)

    micro_meter_size = 501.22
    pixel_over_micro_ratio = 2208 / micro_meter_size

    ground_truth_df = pd.read_csv(corresponding_labels_path, sep="\t")

    if ground_truth_df.empty:
        continue
    if (ground_truth_df["Centroid Y µm"].max() * pixel_over_micro_ratio) > 2208:
        micro_meter_size = 1002.43
        pixel_over_micro_ratio = 2208 / micro_meter_size

    if (ground_truth_df["Centroid Y µm"].max() * pixel_over_micro_ratio) > 2208:
        micro_meter_size = 10024.32
        pixel_over_micro_ratio = 2208 / micro_meter_size
    if "calc_wrong_optovar" in img_path.stem:
        micro_meter_size = 626.52
        pixel_over_micro_ratio = 2208 / micro_meter_size


    # ground_truth_df = ground_truth_df.rename({"class":"class_label", "x": "col", "y": "row"}, axis=1)
    print(ground_truth_df.columns)
    ground_truth_df = ground_truth_df.rename({"Classification":"class_label", "Centroid X µm": "col", "Centroid Y µm": "row"}, axis=1)
    ground_truth_df["row"] =     ground_truth_df["row"] * pixel_over_micro_ratio
    ground_truth_df["col"] =     ground_truth_df["col"] * pixel_over_micro_ratio

    img = convert_czi_to_img(czi)

    if len(img.shape) > 2:
        # Convert rgb to bgr, as the microscrope intended
        img = img[ :, :, ::-1]
    # Normalize between 0 and 1
    img = normImage(img)

    labeled_image = detect_areas_on_rgb_img(img)



    # Get hematoxylin only
    img = extract_hematoxylin(img)
    img = img_as_ubyte(img)

    io.imsave(out_dir_labeled / f"{img_path.stem}_labeled_beforeGroundTruthExtraction.tif", labeled_image, check_contrast=False)
    measured_df = measureLabeledImage(labeled_image)
    measured_df.to_csv(out_dir_measured / "{img_path.stem}_measured_beforeGroundTruthExtraction.csv")

    # plt.imshow(labeled_image)
    # for tmp_row in ground_truth_df.itertuples():
    #     plt.scatter(int(tmp_row.col), int(tmp_row.row), color="red")
    # plt.show()
    # plt.close()


    # first cut the good ones and set their labels to zero in the labeled_image
    cutBboxesFromCenter(ground_truth_df, img, labeled_image, image_prefix = img_path.stem, output_dir = out_dir_objects, mask=False, remove_after = True)

    io.imsave(out_dir_labeled / f"{img_path.stem}_labeled_afterGroundTruthExtraction.tif", labeled_image, check_contrast=False)

    new_measured_df = measureLabeledImage(labeled_image)
    new_measured_df.to_csv(out_dir_measured / f"{img_path.stem}_measured_afterGroundTruthExtraction.csv")
    if len(new_measured_df) > n_keep:
        new_measured_df_subsampled = new_measured_df.sample(n=n_keep)
    else:
        new_measured_df_subsampled = new_measured_df
    new_measured_df_subsampled.to_csv(out_dir_measured / f"{img_path.stem}_measured_afterSubSampling.csv")

    # plt.imshow(labeled_image)
    # for tmp_row in ground_truth_df.itertuples():
    #     plt.scatter(int(tmp_row.col), int(tmp_row.row), color="red")
    # plt.show()
    # plt.close()

    # So now, every label that was even remotely close to the truth is gone, and these labels are also removed from 
    new_measured_df_subsampled = new_measured_df_subsampled.rename({"center_row":"row", "center_col": "col"}, axis=1)
    cutBboxesFromCenter(new_measured_df_subsampled, img, labeled_image, image_prefix = img_path.stem, output_dir = out_dir_objects, mask=False, remove_after = False)


