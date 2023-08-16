import matplotlib.animation as animation
import matplotlib
import matplotlib.pyplot as plt


def save_video(frames, framerate=30, name_file="video.gif"):
    print(frames[0].shape)
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    # Switch to headless 'Agg' to inhibit figure rendering.
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    f = f"results/{name_file}"
    #FFwriter = animation.FFMpegWriter(fps=framerate)
    writergif = animation.PillowWriter(fps=framerate)
    anim.save(f, writer=writergif)
