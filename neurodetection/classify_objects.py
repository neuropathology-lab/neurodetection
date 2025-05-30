from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

def getPatch(arr, idx, radius=3, fill=None):
    """
    Extracts a local patch (neighborhood) around a given index in a NumPy array.

    Parameters:
    arr (ndarray of rank N): Input array.
    idx (tuple): N-dimensional index specifying the center of the patch.
    radius (int or tuple): Number of elements to include on each side along each axis.
    fill (scalar or None): If set, out-of-bounds values are filled with this value. If None, patch is clipped to stay within bounds.

    Returns:
    ndarray: Extracted patch around the specified index.
    """

    assert len(idx) == len(arr.shape)

    if np.isscalar(radius):
        radius = tuple([radius for _ in range(len(arr.shape))])

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

        slices.append(slice(max(0, l), min(arr.shape[axis], r + 1)))
        paddings.append((pl, pr))

    if fill is None:
        return arr[tuple(slices)]
    return np.pad(arr[tuple(slices)], paddings, 'constant', constant_values=fill)

def classifyIsNeuron(objects_df, img, rowname="center_row", colname="center_col", model=None, scaling_factor=1.0):
    """
    Classifies each detected object in the dataframe as a neuron or not,
    using a trained FastAI model and image patches centered on object coordinates.
    """

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

    # Apply scaling to patch radius (assumes 100µm square centered on object)
    scaled_radius = (
        int(50 * scaling_factor),  # row axis
        int(50 * scaling_factor),  # column axis
        None                       # no slicing on channel axis
    )

    for row, col in zip(rows, cols):
        try:
            with tqdm(disable=True):
                # Extract image patch around the object
                img_patch = getPatch(img, (row, col, None), radius=scaled_radius, fill=0)

                # Convert to 8-bit image
                img_patch = (img_patch * 255).astype(np.uint8)

                # Resize to 100x100 pixels as expected by the model
                img_patch_resized = cv2.resize(img_patch, (100, 100), interpolation=cv2.INTER_LINEAR)

                # Run model prediction
                pred_class, pred_idx, outputs = model.predict(img_patch_resized)
                outputs = pd.DataFrame([outputs.tolist()])

                pred_col.append(pred_class)
                prob_col.append(outputs[1])

        except Exception as e:
            tqdm.write(f"Patch at ({row}, {col}) failed: {e}")
            pred_col.append(None)

        bar.update(1)

    bar.close()

    # Store predictions in the DataFrame
    objects_df["is_neuron"] = pred_col
    objects_df["is_neuron_prob"] = prob_col

    return objects_df

def changeSpecificity(neurons_df, min_prob):
    """
    Filters out neurons with prediction probability below the minimum threshold.
    """
    if min_prob != 0.5:
        neurons_df.loc[
            neurons_df['is_neuron_prob'].apply(lambda x: x[0] < min_prob),
            'is_neuron'
        ] = "Negative"
        neurons_df = neurons_df[neurons_df["is_neuron"] == "Positive"]

    return neurons_df