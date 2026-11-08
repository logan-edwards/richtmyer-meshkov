import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_sheet(
    z_data,
    time,
    filename,
    aspect_ratio
    ):
    fig, ax = plt.subplots()

    n_frames = len(z_data)

    sheet_line, = ax.plot(
        np.real(z_data[0]),
        np.imag(z_data[0]),
        'k-'
    )
    
    xmin = min(np.min(np.real(z)) for z in z_data)
    xmax = max(np.max(np.real(z)) for z in z_data)
    ymin = min(np.min(np.imag(z)) for z in z_data)
    ymax = max(np.max(np.imag(z)) for z in z_data)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(aspect_ratio)

    title = ax.set_title(f"t={time[0]:.2f}")

    def update(frame):
        # Each frame may contain a different number of markers.
        sheet_line.set_xdata(np.real(z_data[frame]))
        sheet_line.set_ydata(np.imag(z_data[frame]))

        title.set_text(f"t={time[frame]:.2f}")

        return sheet_line, title

    desired_time = 10  # seconds
    fps_desired = n_frames / desired_time
    interval_desired = desired_time * 1000 / n_frames

    print(f"framerate = {fps_desired} fps")

    sheet_animation = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_desired,
        blit=False
    )

    sheet_animation.save(filename, writer='ffmpeg', fps=fps_desired)

# NOTE the below code is AI-generated...
def plot_side_by_side(
    frames,
    times,
    dash_indices=None,
    x_scale=1.0,
    y_scale=1.0,
    panel_origin=0.0,
    panel_width=1.0,
    colors="black",
    xlabel="",
    ylabel="",
    time_unit="",
    time_scale=1.0,
    equal_aspect=False,
    top_axis=False,
    zero_line=False,
    ylim=None,
    title=None,
    figsize=None,
    save_path=None,
):
    number_frames = len(frames)

    if dash_indices is None:
        dash_indices = [[] for f in range(number_frames)]

    if isinstance(colors, str):
        colors = [colors] * number_frames

    fig, axes = plt.subplots(
        1,
        number_frames,
        sharey=True,
        squeeze=False,
        figsize=figsize,
    )
    axes = axes[0]

    # Make neighboring panels touch.
    fig.subplots_adjust(wspace=0)

    for f in range(number_frames):
        ax = axes[f]
        frame = np.asarray(frames[f])
        color = colors[f]

        x = frame.real / x_scale
        y = frame.imag / y_scale

        # Plot each segment with its requested line style.
        number_segments = len(frame) - 1

        segment_start = 0

        while segment_start < number_segments:
            is_dashed = segment_start in dash_indices[f]
            segment_end = segment_start + 1

            # Extend the group while the line style stays the same.
            while segment_end < number_segments:
                next_is_dashed = segment_end in dash_indices[f]

                if next_is_dashed != is_dashed:
                    break

                segment_end += 1

            if is_dashed:
                line_style = "--"
            else:
                line_style = "-"

            # Include the endpoint of the last segment.
            ax.plot(
                x[segment_start:segment_end + 1],
                y[segment_start:segment_end + 1],
                color=color,
                linestyle=line_style,
            )

            segment_start = segment_end

        ax.set_xlim(panel_origin, panel_origin + panel_width)

        displayed_time = times[f] * time_scale
        time_label = f"$t = {displayed_time:g}$ {time_unit}"

        # Use the title area for each panel's time label.
        ax.set_title(time_label, color=color)

        if zero_line:
            ax.axhline(0, color="black", linestyle="--")

        ax.tick_params(
            direction="in",
            top=True,
            left=(f == 0),
            right=(f == number_frames - 1),
            labelleft=(f == 0),
        )

        # Keep only one spine at each internal boundary.
        if f > 0:
            ax.spines["left"].set_visible(False)

        if top_axis:
            ax.tick_params(
                axis="x",
                labeltop=True,
                labelbottom=False,
            )

        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

    if ylim is not None:
        axes[0].set_ylim(ylim)

    axes[0].set_ylabel(ylabel)

    if top_axis:
        fig.text(0.5, 1.0, xlabel, ha="center", va="bottom")
    else:
        fig.supxlabel(xlabel)

    if title is not None:
        fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes