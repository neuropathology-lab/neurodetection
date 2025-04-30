import pandas as pd
import numpy as np
from skimage import measure
from classify_objects import getPatch
from skimage.color import rgb2hed, hed2rgb, rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from pathlib import Path
from skimage import io

def measureLabeledImage(labeled_image: np.array, original_image: np.array = None, pixels_to_um:int = 0) -> pd.DataFrame:
    regions = measure.regionprops(labeled_image, intensity_image=original_image)

    propList = ['Area',
                'bbox',
                'equivalent_diameter', 
                'orientation', 
                'MajorAxisLength',
                'MinorAxisLength',
                'Perimeter',
                'MinIntensity',
                'MeanIntensity',
                'MaxIntensity']    

    rows_list=[]
    for region_props in regions:
        attribute_dict = {}
        center_row, center_col= region_props['centroid']
        attribute_dict['image_label'] =region_props['Label']
        attribute_dict['cell_label'] = f"row{int(center_row)}_col{int(center_col)}_{region_props['Label']}"
        attribute_dict['center_row'] = int(center_row)
        attribute_dict['center_col'] = int(center_col)
        for i,prop in enumerate(propList):
            if(prop == 'Area') and pixels_to_um != 0: 
                attribute_dict['real_area'] = region_props[prop]*pixels_to_um**2
            elif (prop.find('Intensity') < 0) and pixels_to_um != 0:          # Any prop without Intensity in its name
                attribute_dict[prop] = region_props[prop]*pixels_to_um
            elif (prop.find('Intensity') < 0):
                attribute_dict[prop] = region_props[prop]
            else: 
                if original_image is not None:
                    attribute_dict[prop] = region_props[prop]

        rows_list.append(attribute_dict)
    attribute_df = pd.DataFrame(rows_list)
    return attribute_df

def detect_areas_on_rgb_img(rgb_img):
    gray_img = rgb2gray(rgb_img)

    thresh = threshold_otsu(gray_img)
    # imgs are black on white
    binary = gray_img < thresh
    # eroded = erosion(binary, disk(5))
    filled = binary_fill_holes(binary)
    labeled_image = label(binary)
    return labeled_image

def cutBboxesFromCenter(df: pd.DataFrame, image: np.ndarray, labeled_image: np.ndarray, output_dir: str or Path = None, image_prefix: str = "", mask = True, remove_after = False):
    if image_prefix and not image_prefix.endswith("_"):
        image_prefix += "_"
    if not output_dir:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    for row in df.itertuples():
        row = row._asdict()
        image_label = labeled_image[round(row["row"]), round(row["col"])]
        cell_name = f"row{round(row['row'])}col{round(row['col'])}label{image_label}"
        bb = getPatch(image, (round(row["row"]), round(row["col"]), None), radius=(50, 50, None), fill=None).copy()
        if remove_after:
            labeled_bb = getPatch(labeled_image, (round(row["row"]), round(row["col"])), radius=(50, 50), fill=None)
        else:
            labeled_bb =  getPatch(labeled_image, (round(row["row"]), round(row["col"])), radius=(50, 50), fill=None).copy()

        if mask:
            bb[labeled_bb != row["image_label"]] = 0

        if remove_after:
            io.imsave(output_dir / f"{image_prefix}cell_{cell_name}_label{row['class_label']}.png", bb, check_contrast=False)
            labels_to_remove_in_this_bb = np.unique(labeled_bb)
            for tmp_label in labels_to_remove_in_this_bb:
                labeled_bb[labeled_bb == tmp_label] = 0
        else:
            io.imsave(output_dir / f"{image_prefix}cell_{cell_name}_labelNegative.png", bb, check_contrast=False)
