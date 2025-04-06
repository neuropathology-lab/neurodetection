import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.ioff()

def threePlotsSave(img, objects_df, neurons_df, output_path_plots,
                   square_size, pixel_size, edge_threshold_pixels, plot_type, max_dim=10):
    tab10 = plt.get_cmap('tab10').colors

    neurons = neurons_df[(neurons_df["close_objects"] == False) & (neurons_df["objects_edges"] == False)]
    neurons_discarded = neurons_df[~neurons_df.index.isin(neurons.index)]

    img_height, img_width = img.shape[:2]
    aspect_ratio = img_width / img_height
    square_size_pixels = square_size / pixel_size
    half_square = square_size_pixels / 2

    def format_axes(ax, title):
        ax.imshow(img)
        ax.set_title(title, fontsize=max_dim * 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # origin='upper'

    def draw_squares(ax, df, color):
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
        # Bottom-right positioning with margin
        margin_x = 0.05 * img_width
        margin_y = 0.05 * img_height
        square_size_px = square_size_pixels

        square_x = img_width - margin_x - square_size_px
        square_y = img_height - margin_y - square_size_px

        # Draw filled orange square with black border
        ax.add_patch(Rectangle(
            (square_x, square_y),
            square_size_px, square_size_px,
            linewidth=1,
            edgecolor='black',
            facecolor='orange',
            alpha=0.5
        ))

        # Label above square in bold black font
        ax.text(
            square_x + square_size_px / 2,
            square_y - 0.01 * img_height,
            f"{square_size:.1f} µm",
            color='black',
            ha='center',
            va='bottom',
            fontsize=max_dim*2)

    if plot_type == "detailed":
        subplot_width = min(max_dim, max_dim * aspect_ratio)
        subplot_height = min(max_dim, max_dim / aspect_ratio)
        fig, axs = plt.subplots(2, 2, figsize=(2 * subplot_width, 2 * subplot_height), constrained_layout=True)
        axs = axs.flatten()

        titles = ["Original image (with reference classification square)", "All detected objects", "Objects classified as neurons", " Final neuron detections"]
        for i, title in enumerate(titles):
            format_axes(axs[i], title)

        # Clip scatter points to within image bounds
        in_bounds = (
            (objects_df["center_col"] >= 0) & (objects_df["center_col"] < img_width) &
            (objects_df["center_row"] >= 0) & (objects_df["center_row"] < img_height)
        )
        objects_df_clipped = objects_df[in_bounds]

        axs[1].scatter(
            objects_df_clipped["center_col"],
            objects_df_clipped["center_row"],
            color="FireBrick",
            s=(max_dim * max_dim) / 2
        )

        for i in [2, 3]:
            draw_squares(axs[i], neurons, tab10[1])
        draw_squares(axs[2], neurons_discarded, 'yellow')

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

        # Add reference square in first subplot
        draw_reference_square(axs[0])

    elif plot_type == "simple":
        fig_width = max_dim if aspect_ratio >= 1 else max_dim * aspect_ratio
        fig_height = max_dim / aspect_ratio if aspect_ratio >= 1 else max_dim
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        format_axes(ax, "Detected neurons")
        draw_squares(ax, neurons, tab10[1])

    fig.savefig(output_path_plots)
    plt.close(fig)