from .utils import getPatch
from .load_model import load_is_neuron, load_neuron_subtype
import numpy as np
from tqdm import tqdm

def classify_is_neuron(measured_df, img, rowname="center_row", colname="center_col", model=None):
    import numpy as np

    if model is None:
        model = load_is_neuron(model)
        print("Model detect neurons is none")

    pred_col = []

    # Use .loc access for speed
    rows = measured_df[rowname].round().astype(int).values
    cols = measured_df[colname].round().astype(int).values

    # Create one overall progress bar
    progress_bar = tqdm(total=len(rows), desc="Processing: ", leave=True, mininterval = 0.0001, maxinterval = 10)

    for row, col in zip(rows, cols):
        try:
            img_patch = getPatch(img, (row, col, None), radius=(50, 50, None), fill=0)
            if img_patch.ndim == 2:  # Convert grayscale to RGB
                img_patch = np.stack([img_patch] * 3, axis=-1)
            img_patch = (img_patch * 255).astype(np.uint8)
            pred = model.predict(img_patch)
            pred_col.append(pred[0])
        except Exception as e:
            print(f"Patch at ({row}, {col}) failed: {e}")
            pred_col.append(None)

        progress_bar.update(1)  # Update the overall progress bar by one step

    progress_bar.close()
    measured_df["is_neuron"] = pred_col
    return measured_df

def classify_neuron_subtype(measured_df, img, rowname="center_row", colname="center_col", model=None):
    if model is None:
        model = load_neuron_subtype(model)
        print("Model classify subtype is none")
    pred_col = []
    for tmp_row in measured_df.itertuples():
        tmp_row = tmp_row._asdict()
        if tmp_row["is_neuron"] != "Negative":
            img_patch = getPatch(img, (round(tmp_row[rowname]), round(tmp_row[colname]), None), radius=(50,50,None), fill=None).copy()
            img_patch = (img_patch * 255).astype(np.uint8)
            pred = model.predict(img_patch)

            pred_col.append(pred[0])
        else:
            pred_col.append("Negative")

    measured_df["neuron_subtype"] = pred_col

    return measured_df
