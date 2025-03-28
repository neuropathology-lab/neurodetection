from .utils import getPatch
from tqdm import tqdm
import numpy as np

def classify_is_neuron(measured_df, img, rowname="center_row", colname="center_col", model=None):
    pred_col = []

    # Use .loc access for speed
    rows = measured_df[rowname].round().astype(int).values
    cols = measured_df[colname].round().astype(int).values

    bar = tqdm(
        total=len(rows),
        desc="Classifying neurons",
        dynamic_ncols=True,
        mininterval=0.05,
        maxinterval=0.5,
        ascii=True
    )

    for row, col in zip(rows, cols):
        try:
            with tqdm(disable=True):
                img_patch = getPatch(img, (row, col, None), radius=(50, 50, None), fill=0)
                if img_patch.ndim == 2:
                    img_patch = np.stack([img_patch] * 3, axis=-1)
                img_patch = (img_patch * 255).astype(np.uint8)

                pred = model.predict(img_patch)
            pred_col.append(pred[0])
        except Exception as e:
            tqdm.write(f"Patch at ({row}, {col}) failed: {e}")
            pred_col.append(None)

        bar.update(1)

    bar.close()

    measured_df["is_neuron"] = pred_col

    return measured_df