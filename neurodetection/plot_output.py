import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.ioff()  # Turn interactive mode off to prevent auto-plotting in some environments

def threePlotsSave(org_img, img, objects_df, neurons_df, output_path_plots,
                   square_size, pixel_size, edge_threshold_pixels, plot_type, use_hematoxylin, max_dim=10):
    """
    Plot types:

    detailed – Four subplots:
        1) Original image with a representation of the square size used for classification.
        2) All detected objects.
        3) All objects classified as neurons, including those to be discarded (in yellow).
        4) Final neuron detections, which will be saved and used to calculate density.

    simple – One plot showing only the final neuron detections.

    no_neurons – Two plots, saved when no object was classified as a neuron:
        1) Original image with a representation of the square size used for classification.
        2) All detected objects.
    """

    # Convert original and processed image (if was not converted to hematoxylin channel)
    # from BGR to RGB for plotting
    org_img = org_img[:, :, ::-1]

    if not use_hematoxylin:
        img = img[:, :, ::-1]

    tab10 = plt.get_cmap('tab10').colors  # Color palette

    if plot_type != "no_neurons":
        # Keep only valid neurons (not too close and not on edges)
        neurons = neurons_df[(neurons_df["close_objects"] == False) & (neurons_df["objects_edges"] == False)]
        # Neurons that were discarded (either too close or on edge)
        neurons_discarded = neurons_df[~neurons_df.index.isin(neurons.index)]

    # Get image dimensions and derived measurements
    img_height, img_width = img.shape[:2]
    aspect_ratio = img_width / img_height
    square_size_pixels = square_size / pixel_size
    half_square = square_size_pixels / 2

    def format_axes(ax, title, img):
        """
        Display image with given formatting.
        """
        ax.imshow(img)
        ax.set_title(title, fontsize=max_dim * 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Flip y-axis to match image origin (top-left)

    def draw_squares(ax, df, color):
        """
        Draw classification square at each object location.
        """
        for _, row in df.iterrows():
            rect = Rectangle(
                (row["center_col"] - half_square, row["center_row"] - half_square),
                square_size_pixels, square_size_pixels,
                linewidth=max_dim / 4,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

    def draw_reference_square(ax):
        """
        Draw a labeled reference square in the bottom-right corner of the image.
        """
        margin_x = 0.05 * img_width
        margin_y = 0.05 * img_height
        square_size_px = square_size_pixels

        square_x = img_width - margin_x - square_size_px
        square_y = img_height - margin_y - square_size_px

        ax.add_patch(Rectangle(
            (square_x, square_y),
            square_size_px, square_size_px,
            linewidth=1,
            edgecolor='black',
            facecolor='orange',
            alpha=0.5
        ))

        ax.text(
            square_x + square_size_px / 2,
            square_y - 0.01 * img_height,
            f"{square_size:.1f} µm",
            color='black',
            ha='center',
            va='bottom',
            fontsize=max_dim * 2
        )

    if plot_type == "detailed":
        # Create a 2x2 grid of subplots
        subplot_width = min(max_dim, max_dim * aspect_ratio)
        subplot_height = min(max_dim, max_dim / aspect_ratio)
        fig, axs = plt.subplots(2, 2, figsize=(2 * subplot_width, 2 * subplot_height), constrained_layout=True)
        axs = axs.flatten()

        titles = ["Original image (with reference classification square)",
                  "All detected objects",
                  "Objects classified as neurons",
                  "Final neuron detections"]

        for i, title in enumerate(titles):
            if i == 0:
                format_axes(axs[i], title, org_img)
            else:
                format_axes(axs[i], title, img)

        # Clip points to within image boundaries
        in_bounds = (
                (objects_df["center_col"] >= 0) & (objects_df["center_col"] < img_width) &
                (objects_df["center_row"] >= 0) & (objects_df["center_row"] < img_height)
        )
        objects_df_clipped = objects_df[in_bounds]

        # Draw reference square on original image
        draw_reference_square(axs[0])

        # Plot all detected object centers
        axs[1].scatter(
            objects_df_clipped["center_col"],
            objects_df_clipped["center_row"],
            color="FireBrick",
            s=(max_dim * max_dim) / 2
        )

        # Draw neuron squares
        for i in [2, 3]:
            draw_squares(axs[i], neurons, tab10[1])  # blue squares
        draw_squares(axs[2], neurons_discarded, 'yellow')  # discarded in yellow

        # Draw dashed border indicating exclusion zone near edges
        edge_frame = Rectangle(
            (edge_threshold_pixels, edge_threshold_pixels),
            img_width - 2 * edge_threshold_pixels,
            img_height - 2 * edge_threshold_pixels,
            linewidth=2,
            edgecolor='black',
            linestyle=':',
            facecolor='none'
        )
        axs[2].add_patch(edge_frame)

    elif plot_type == "simple":
        # Single plot with final neuron detections
        fig_width = max_dim if aspect_ratio >= 1 else max_dim * aspect_ratio
        fig_height = max_dim / aspect_ratio if aspect_ratio >= 1 else max_dim
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        format_axes(ax, "Detected neurons", org_img)
        draw_squares(ax, neurons, tab10[1])  # blue squares

    elif plot_type == "no_neurons":
        # Two-panel plot: original image and all detected objects
        subplot_width = min(max_dim, max_dim * aspect_ratio)
        subplot_height = min(max_dim, max_dim / aspect_ratio)
        fig, axs = plt.subplots(1, 2, figsize=(2 * subplot_width, subplot_height), constrained_layout=True)
        axs = axs.flatten()

        titles = ["Original image (with reference classification square)", "All detected objects"]
        for i, title in enumerate(titles):
            if i == 0:
                format_axes(axs[i], title, org_img)
            else:
                format_axes(axs[i], title, img)

        # Clip points to within image bounds
        in_bounds = (
                (objects_df["center_col"] >= 0) & (objects_df["center_col"] < img_width) &
                (objects_df["center_row"] >= 0) & (objects_df["center_row"] < img_height)
        )
        objects_df_clipped = objects_df[in_bounds]

        # Draw reference square on original image
        draw_reference_square(axs[0])

        # Plot all detected object centers
        axs[1].scatter(
            objects_df_clipped["center_col"],
            objects_df_clipped["center_row"],
            color="FireBrick",
            s=(max_dim * max_dim) / 2
        )

    # Save the figure to the specified output path
    fig.savefig(output_path_plots)
    plt.close(fig)