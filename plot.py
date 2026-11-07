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

def plot_side_by_side(
    frames,
    times,
    x_scale=1.0,
    y_scale=1.0,
    panel_origin=0.0,
    panel_width=1.0,
    colors=None,
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
    """
    Plot complex-valued snapshots in adjacent horizontal panels.

    Each frame is a 1D complex array:
        real(frame) / x_scale -> local horizontal coordinate
        imag(frame) / y_scale -> vertical coordinate

    Each successive frame is shifted right by panel_width.

    panel_origin and panel_width are in scaled horizontal coordinates.
    Frames may contain different numbers of points.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    frames = list(frames)
    times = np.asarray(times)

    if len(frames) == 0:
        raise ValueError("len(frames) = 0")
    if times.ndim != 1 or len(frames) != len(times):
        raise ValueError("data does not match up")
    if x_scale <= 0 or y_scale <= 0 or panel_width <= 0:
        raise ValueError("negative geometric scales provided")

    n_frames = len(frames)

    if colors is None:
        default_colors = [
            "blue", "orangered", "green", "crimson", "navy"
        ]
        colors = [
            default_colors[i % len(default_colors)]
            for i in range(n_frames)
        ]
    else:
        colors = list(colors)
        if len(colors) != n_frames:
            raise ValueError("Provide one color per frame.")

    if figsize is None:
        figsize = (2.4 * n_frames, 3.2)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    left = panel_origin
    right = left + n_frames * panel_width

    # Internal panel boundaries.
    for i in range(1, n_frames):
        boundary = left + i * panel_width
        ax.axvline(
            boundary,
            color="0.35",
            linewidth=0.8,
            linestyle=(0, (8, 6)),
            zorder=0,
        )

    if zero_line:
        ax.axhline(
            0.0,
            color="0.35",
            linewidth=0.8,
            linestyle=(0, (8, 6)),
            zorder=0,
        )

    for i, (frame, time, color) in enumerate(
        zip(frames, times, colors)
    ):
        frame = np.asarray(frame)

        if frame.ndim != 1:
            raise ValueError("Each frame must be a 1D array.")

        # Preserve point ordering along the sheet.
        x_plot = frame.real / x_scale + i * panel_width
        y_plot = frame.imag / y_scale

        ax.plot(x_plot, y_plot, color=color, linewidth=1.1)

        # Horizontal position in data coordinates;
        # vertical position as a fraction of axes height.
        label_x = left + (i + 0.5) * panel_width
        displayed_time = time * time_scale
        label = f"$t = {displayed_time:g}$"
        if time_unit:
            label += f" {time_unit}"

        ax.text(
            label_x,
            0.96,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=color,
            fontsize=12,
        )

    ax.set_xlim(left, right)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.margins(y=0.18)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.tick_params(
        axis="both",
        direction="in",
        top=True,
        right=True,
    )

    if top_axis:
        ax.xaxis.set_label_position("top")
        ax.tick_params(
            axis="x",
            labeltop=True,
            labelbottom=False,
        )

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax