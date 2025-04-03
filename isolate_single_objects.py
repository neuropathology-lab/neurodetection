import numpy as np
import pandas as pd
from skimage import io
from typing import List
from pathlib import Path

def cut_bboxes_from_center(df: pd.DataFrame, image: np.ndarray, labeled_image: np.ndarray, output_dir: str or Path = None, image_prefix: str = "", mask = True, remove_after = False) -> List:

    if image_prefix and not image_prefix.endswith("_"):
        image_prefix += "_"
    if not output_dir:
        output_dir = Path("neurodetection")
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
            io.imsave(output_dir / f"{image_prefix}cell_{cell_name}_label{row['class_label']}.tif", bb, check_contrast=False)
            labels_to_remove_in_this_bb = np.unique(labeled_bb)
            for tmp_label in labels_to_remove_in_this_bb:
                labeled_bb[labeled_bb == tmp_label] = 0
        else:
            io.imsave(output_dir / f"{image_prefix}cell_{cell_name}_labelNegative.tif", bb, check_contrast=False)