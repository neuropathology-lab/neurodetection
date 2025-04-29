from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

def getPatch(arr, idx, radius=3, fill=None):
    """
    Gets surrounding elements from a numpy array

    Parameters:
    arr (ndarray of rank N): Input array
    idx (N-Dimensional Index): The index at which to get surrounding elements. If None is specified for a particular axis,
        the entire axis is returned.
    radius (array-like of rank N or scalar): The radius across each axis. If None is specified for a particular axis,
        the entire axis is returned.
    fill (scalar or None): The value to fill the array for indices that are out-of-bounds.
        If value is None, only the surrounding indices that are within the original array are returned.

    Returns:
    ndarray: The surrounding elements at the specified index
    """

    assert len(idx) == len(arr.shape)

    if np.isscalar(radius):
        radius = tuple([radius for i in range(len(arr.shape))])

    slices = []
    paddings = []
    for axis in range(len(arr.shape)):
        if idx[axis] is None or radius[axis] is None:
            slices.append(slice(0, arr.shape[axis]))
            paddings.append((0, 0))
            continue

        r = radius[axis]
        l = idx[axis] - r
        r = idx[axis] + r

        pl = 0 if l > 0 else abs(l)
        pr = 0 if r < arr.shape[axis] else r - arr.shape[axis] + 1

        slices.append(slice(max(0, l), min(arr.shape[axis], r+1)))
        paddings.append((pl, pr))

    if fill is None:
        return arr[tuple(slices)]
    return np.pad(arr[tuple(slices)], paddings, 'constant', constant_values=fill)

def classifyIsNeuron(objects_df, img, rowname="center_row", colname="center_col", model=None, scaling_factor=1.0):

    pred_col = []
    prob_col = []

    # Use .loc access for speed
    rows = objects_df[rowname].round().astype(int).values
    cols = objects_df[colname].round().astype(int).values

    bar = tqdm(
        total=len(rows),
        desc="Classifying neurons",
        dynamic_ncols=True,
        mininterval=0.05,
        maxinterval=0.5,
        ascii=True
    )
    # Apply scaling to radius
    scaled_radius = (
        int(50 * scaling_factor),
        int(50 * scaling_factor),
        None
    )
    for row, col in zip(rows, cols):
        try:
            with tqdm(disable=True):

                img_patch = getPatch(img, (row, col, None), radius=scaled_radius, fill=0)

                img_patch = (img_patch * 255).astype(np.uint8)

                # Resize to 100x100 pixels
                img_patch_resized = cv2.resize(img_patch, (100, 100), interpolation=cv2.INTER_LINEAR)

                # Classify
                pred_class, pred_idx, outputs = model.predict(img_patch_resized)
                outputs = pd.DataFrame([outputs.tolist()])

                pred_col.append(pred_class)
                prob_col.append(outputs[1])

        except Exception as e:
            tqdm.write(f"Patch at ({row}, {col}) failed: {e}")
            pred_col.append(None)

        bar.update(1)

    bar.close()

    objects_df["is_neuron"] = pred_col
    objects_df["is_neuron_prob"] = prob_col

    return objects_df

def changeSpecificity(neurons_df, min_prob):
    if min_prob != 0.5:

        neurons_df.loc[
            neurons_df['is_neuron_prob'].apply(lambda x: x[0] < min_prob),
            'is_neuron'
        ] = "Negative"
        neurons_df = neurons_df[neurons_df["is_neuron"] == "Positive"]

    return neurons_df