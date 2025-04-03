import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_traffic(traffic_density):
    intersections = ["A", "B", "C", "D"]
    fig, ax = plt.subplots()
    bars = ax.bar(intersections, traffic_density, color="blue")

    # Add gridlines for a professional look
    ax.grid(True)

    def update(frame):
        # Simulate dynamic traffic density updates
        new_density = np.random.randint(0, 100, size=len(intersections))
        for bar, height in zip(bars, new_density):
            bar.set_height(height)
        ax.set_title(f"Traffic Density at Intersections (Frame {frame})")

    ani = animation.FuncAnimation(fig, update, frames=20, interval=500, repeat=False)
    plt.xlabel("Intersections")
    plt.ylabel("Traffic Density")
    plt.title("Traffic Density at Intersections")

    # Display the animation
    plt.show()