import matplotlib.pyplot as plt

def three_plots_save(img, objects_df, neurons_df, output_plot_path, neuron_points_size):

    tab10 = plt.get_cmap('tab10').colors
    plt.rcParams["figure.figsize"] = [30, 9]
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    axs = axs.flatten()

    # plot with only raw photo
    axs[0].imshow(img)
    axs[0].set_title("Raw photo", fontsize=20)
    # plot with all detected objects
    axs[1].imshow(img)
    axs[1].set_title("All detected objects", fontsize=20)
    axs[1].scatter(objects_df["center_col"], objects_df["center_row"], color="r")
    # plot with detected neurons
    axs[2].imshow(img)
    axs[2].set_title("Detected neurons", fontsize=20)
    axs[2].scatter(
        neurons_df["center_col"],
        neurons_df["center_row"],
        color=tab10[1], facecolors='none', edgecolors=tab10[1],
        marker='o', s=neuron_points_size, linewidths=2)

    # Save plot
    fig.savefig(output_plot_path)
    plt.close(fig)