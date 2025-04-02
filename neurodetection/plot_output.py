import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.ioff()

def three_plots_save(img, objects_df, neurons_df, output_plot_path, square_size_pixels, edge_threshold_pixels, plot_type = "detailed", max_dim=10):
    tab10 = plt.get_cmap('tab10').colors

    # Prepare neuron subsets
    neurons = neurons_df[(neurons_df["close_objects"] == False) & (neurons_df["objects_edges"] == False)]
    neurons_edge = neurons_df[(neurons_df["close_objects"] == True) | (neurons_df["objects_edges"] == True)]

    img_height, img_width = img.shape[:2]
    aspect_ratio = img_width / img_height

    half_square = square_size_pixels / 2

    if plot_type == "detailed":

        # Use this size for all subplots for consistency
        subplot_width = min(max_dim, max_dim * aspect_ratio)
        subplot_height = min(max_dim, max_dim / aspect_ratio)

        # Final figure size: 2x2 subplots → 2 rows and 2 cols
        fig_width = 2 * subplot_width
        fig_height = 2 * subplot_height

        plt.rcParams["figure.figsize"] = [fig_width, fig_height]

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)
        axs = axs.flatten()

        # plot with only raw photo
        axs[0].imshow(img)
        axs[0].set_title("Raw photo", fontsize=max_dim*2)
        axs[0].tick_params(axis='both', which='major', labelsize=max_dim)

        # plot with all detected objects
        axs[1].imshow(img)
        axs[1].set_title("All detected objects", fontsize=max_dim*2)
        axs[1].scatter(objects_df["center_col"], objects_df["center_row"], color="r", s = (max_dim*max_dim)/2)
        axs[1].tick_params(axis='both', which='major', labelsize=max_dim)

        # Plot with dynamic-size squares for neurons
        axs[2].imshow(img)
        axs[2].set_title("All detected neurons",
                         fontsize=max_dim*2)
        axs[2].tick_params(axis='both', which='major', labelsize=max_dim)


        # Draw orange squares for far neurons
        for _, row in neurons.iterrows():
            rect = Rectangle(
                (row["center_col"] - half_square, row["center_row"] - half_square),
                square_size_pixels, square_size_pixels,
                linewidth=max_dim/4,
                edgecolor=tab10[1],
                facecolor='none'
            )
            axs[2].add_patch(rect)

        # Draw light grey squares for edge neurons
        for _, row in neurons_edge.iterrows():
            rect = Rectangle(
                (row["center_col"] - half_square, row["center_row"] - half_square),
                square_size_pixels, square_size_pixels,
                linewidth=max_dim/4,
                edgecolor='yellow',
                facecolor='none'
            )
            axs[2].add_patch(rect)

        # Add dotted frame indicating edge threshold
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

        # Plot with same dynamic-size squares for comparison
        axs[3].imshow(img)
        axs[3].set_title("Detected neurons (final)", fontsize=max_dim*2)

        for _, row in neurons.iterrows():
            rect = Rectangle(
                (row["center_col"] - half_square, row["center_row"] - half_square),
                square_size_pixels, square_size_pixels,
                linewidth=max_dim/4,
                edgecolor=tab10[1],
                facecolor='none'
            )
            axs[3].add_patch(rect)

    if plot_type == "simple":
        # Simpler plot with capped size based on image shape
        max_dim = 10
        if aspect_ratio >= 1:
            fig_width = max_dim
            fig_height = max_dim / aspect_ratio
        else:
            fig_width = max_dim * aspect_ratio
            fig_height = max_dim

        plt.rcParams["figure.figsize"] = [fig_width, fig_height]

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title("Detected neurons", fontsize=max_dim*2)
        ax.tick_params(axis='both', which='major', labelsize=max_dim)

        for _, row in neurons.iterrows():
            rect = Rectangle(
                (row["center_col"] - half_square, row["center_row"] - half_square),
                square_size_pixels, square_size_pixels,
                linewidth=max_dim/4,
                edgecolor=tab10[1],
                facecolor='none'
            )
            ax.add_patch(rect)

    # Save plot
    fig.savefig(output_plot_path)
    plt.close(fig)