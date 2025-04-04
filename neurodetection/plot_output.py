import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FuncFormatter
import math

plt.ioff()

def threePlotsSave(img, objects_df, neurons_df, output_plot_path, square_size, pixel_size, edge_threshold_pixels, plot_type, max_dim):
    tab10 = plt.get_cmap('tab10').colors

    # Split neuron categories
    neurons = neurons_df[~(neurons_df["close_objects"] | neurons_df["objects_edges"])]
    neurons_edge = neurons_df[neurons_df["close_objects"] | neurons_df["objects_edges"]]

    img_height, img_width = img.shape[:2]
    aspect_ratio = img_width / img_height
    square_size_pixels = square_size / pixel_size
    half_square = square_size_pixels / 2

    # Tick spacing in micrometers
    def roundup(x):
        return int(math.floor(x / 10.0)) * 10

    tick_spacing_height_um = roundup(img_height*pixel_size/5)
    tick_spacing_width_um = roundup(img_height*pixel_size/5)

    # Convert that spacing to pixels
    tick_spacing_height_px = tick_spacing_height_um / pixel_size
    tick_spacing_width_px = tick_spacing_width_um / pixel_size

    formatter = FuncFormatter(lambda value, _: f'{value * pixel_size:.0f}')

    def format_axes(ax, title):
        ax.imshow(img)
        ax.set_title(title, fontsize=max_dim*2)
        ax.yaxis.set_major_locator(MultipleLocator(tick_spacing_height_px))
        ax.xaxis.set_major_locator(MultipleLocator(tick_spacing_width_px))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', which='major', labelsize=max_dim*1.5)

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

    if plot_type == "detailed":
        subplot_width = min(max_dim, max_dim * aspect_ratio)
        subplot_height = min(max_dim, max_dim / aspect_ratio)
        fig, axs = plt.subplots(2, 2, figsize=(2 * subplot_width, 2 * subplot_height), sharex=True, sharey=True, constrained_layout=True)
        axs = axs.flatten()

        titles = ["Raw photo", "All detected objects", "All detected neurons", "Detected neurons (final)"]
        for i, title in enumerate(titles):
            format_axes(axs[i], title)

        axs[1].scatter(objects_df["center_col"], objects_df["center_row"], color="r", s=(max_dim * max_dim) / 2)

        # For axs[2] and axs[3], draw squares
        for i in [2, 3]:
            draw_squares(axs[i], neurons, tab10[1])
        draw_squares(axs[2], neurons_edge, 'yellow')

        # Add edge threshold frame to axs[2]
        frame = Rectangle(
            (edge_threshold_pixels, edge_threshold_pixels),
            img_width - 2 * edge_threshold_pixels,
            img_height - 2 * edge_threshold_pixels,
            linewidth=2,
            edgecolor='black',
            linestyle=':',
            facecolor='none'
        )
        axs[2].add_patch(frame)

    elif plot_type == "simple":
        max_dim = 10
        fig_width = max_dim if aspect_ratio >= 1 else max_dim * aspect_ratio
        fig_height = max_dim / aspect_ratio if aspect_ratio >= 1 else max_dim

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        format_axes(ax, "Detected neurons")
        draw_squares(ax, neurons, tab10[1])

    fig.savefig(output_plot_path)
    plt.close(fig)