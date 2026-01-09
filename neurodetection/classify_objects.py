import numpy as np
import cv2

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

def classifyIsNeuron(
    objects_df,
    img,
    rowname="center_row",
    colname="center_col",
    model=None,
    scaling_factor=1.0,
    patch_half_size_px=50,     # base half-size before scaling (100x100 patch)
    out_size=(100, 100),
    print_progress=True,
    log_errors=True,
):
    """
    Classify each object as neuron or not using a FastAI model on centered image patches.

    Prints only:
      - start
      - 50%
      - 100%

    Expects:
      - objects_df has columns rowname/colname
      - getPatch(img, center=(row,col,None), radius=(r,c,None), fill=0) exists
      - model.predict(image) returns (pred_class, pred_idx, outputs)
    """

    if model is None:
        raise ValueError("model must be provided")

    # Precompute integer coordinates (fast)
    rows = objects_df[rowname].to_numpy(dtype=float)
    cols = objects_df[colname].to_numpy(dtype=float)
    rows = np.rint(rows).astype(np.int32)
    cols = np.rint(cols).astype(np.int32)

    n = len(rows)
    if n == 0:
        # Ensure columns exist even if empty input
        objects_df["is_neuron"] = []
        objects_df["is_neuron_prob"] = []
        return objects_df

    # Patch radius after scaling (keep consistent with your getPatch API)
    r = int(round(patch_half_size_px * scaling_factor))
    scaled_radius = (r, r, None)

    pred_col = [None] * n
    prob_col = [None] * n

    # Progress markers
    half_idx = n // 2  # print when we've *completed* half_idx items
    if print_progress:
        print(f"Classifying neurons: 0% (0/{n})")

    for i, (row, col) in enumerate(zip(rows, cols), start=1):
        try:
            img_patch = getPatch(img, (row, col, None), radius=scaled_radius, fill=0)

            # Convert to 8-bit (assumes img_patch in [0,1]; if not, adjust upstream)
            img_patch_u8 = (img_patch * 255).astype(np.uint8)

            # Resize to model input
            img_patch_resized = cv2.resize(img_patch_u8, out_size, interpolation=cv2.INTER_LINEAR)

            pred_class, pred_idx, outputs = model.predict(img_patch_resized)

            # Extract probability for class index 1 (same semantics as your original code)
            # outputs may be torch tensor / numpy array / list
            out_arr = np.asarray(outputs)
            prob = float(out_arr[1]) if out_arr.size > 1 else float(out_arr.squeeze())

            pred_col[i - 1] = pred_class
            prob_col[i - 1] = prob

        except Exception as e:
            pred_col[i - 1] = None
            prob_col[i - 1] = None
            if log_errors:
                print(f"Patch at ({row}, {col}) failed: {e}")

        # Print only at 50% and 100%
        if print_progress:
            if i == half_idx:
                print(f"Classifying neurons: 50% ({i}/{n})")
            elif i == n:
                print(f"Classifying neurons: 100% ({i}/{n})")

    objects_df = objects_df.copy()
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