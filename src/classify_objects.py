import numpy as np
from tqdm import tqdm
from .utils import getPatch
from .load_model import load_is_neuron, load_neuron_subtype

def classify_is_neuron(measured_df, img, rowname="center_row", colname="center_col", model=None):
    if model is None:
        model = load_is_neuron(model)
        print("Model detect neurons is none")
        
    pred_col = []
    for tmp_row in measured_df.itertuples():
        tmp_row = tmp_row._asdict()
        img_patch = getPatch(img, (round(tmp_row[rowname]), round(tmp_row[colname]), None), radius=(50,50,None), fill=None).copy()
        img_patch = (img_patch * 255).astype(np.uint8)
        pred = model.predict(img_patch)
        pred_col.append(pred[0])
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
