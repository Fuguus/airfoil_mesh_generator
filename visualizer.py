import matplotlib.pyplot as plt

def visualize_points(x, y):
    """
    Display a scatter plot of points (x, y).
    
    Parameters:
    x (array-like): X coordinates of the points.
    y (array-like): Y coordinates of the points.
    """
    plt.scatter(x, y, color='blue')
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def visualize_graph(x, y, title, x_label, y_label):
    """
    Plot a 2D line graph of y versus x with labeled axes and a title.
    
    Parameters:
    x (array-like): X-axis values.
    y (array-like): Y-axis values.
    title (str): Title of the graph.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    """
    plt.plot(x, y, color='blue')
    plt.axis("equal")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def visualize_points_ylog(x, y, title, x_label, y_label):
    """
    Plot a semilog graph with a logarithmic y-axis.
    
    Parameters:
    x (array-like): X-axis values.
    y (array-like): Y-axis values.
    title (str): Title of the graph.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    """
    plt.figure()
    plt.semilogy(x, y, color='blue')  # Log scale for y-axis
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, which="both", ls="--")
    plt.show()


def plot_grid(x, y, title):
    """
    Plot a 2D grid based on x and y mesh arrays.
    
    Parameters:
    x (ndarray): 2D array of x coordinates (meshgrid).
    y (ndarray): 2D array of y coordinates (meshgrid).
    title (str): Title of the plot.
    """
    for j in range(y.shape[1]):
        plt.plot(x[:, j], y[:, j], 'b-', linewidth=0.5)  # Vertical grid lines
    for i in range(x.shape[0]):
        plt.plot(x[i, :], y[i, :], 'r-', linewidth=0.5)  # Horizontal grid lines
    plt.axis("equal")
    plt.title(title)
    plt.grid(True)
    plt.show()
