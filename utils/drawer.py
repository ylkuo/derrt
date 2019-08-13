import matplotlib.pyplot as plt
import numpy as np
import torchvision

from descartes import PolygonPatch

def plot_line(ax, line, linewidth=1, color='gray'):
    """Plot a line on a given matplotlib axis.

    ax -- the target matplotlib axis
    line -- the shaply LineString to be drawn
    """
    x, y = line.xy
    ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=1)

def plot_poly(ax, poly, color, alpha=1.0, zorder=1):
    """Plot a polygon on a given matplotlib axis.

    ax -- the target matplotlib axis
    poly -- the PolygonPatch to be drawn
    color -- the matplotlib color string of the polygon
    alpha -- the alpha value for blending the color (default 1.0)
    zorder -- the zorder of the polygon, the lower values are drawn first (default 1)
    """
    patch = PolygonPatch(poly, fc=color, ec="black", alpha=alpha, zorder=zorder)
    ax.add_patch(patch)

def make_filter_image(layer, use_color=True, scale_each=True):
    """Build an image of the weights of the filters in a given convolutional layer."""
    weights = layer.weight.data.to("cpu")
    if not use_color:
        n_input_channels = weights.size()[1]
        weights = weights.view([weights.size()[0], 1, weights.size()[1]*weights.size()[2], weights.size()[3]])
    img = torchvision.utils.make_grid(weights, normalize=True, scale_each=scale_each)
    return img

def imshow(img, ax=None):
    """Plot a PyTorch image using matplotlib"""
    npimg = img.numpy()
    if ax is None:
        fig, ax = plt.subplots()
    npimg = np.transpose(npimg, (1,2,0)) # this is needed as torch uses a (channel, height, width) representation while matplotlib usees (width, height, channel)

    npimg = (npimg - npimg.min(axis=(0,1)))/(npimg.max(axis=(0,1))-npimg.min(axis=(0,1))) # map the values to the interval [0,1] for each channel
    ax.imshow(npimg)
    ax.grid(False)

def visualize_filters(layer, use_color=True, scale_each=True):
    """Plot the weights of the filters in a given convolutional layer.
   
    If use_color is true (default), the input layer is expected to have either 1 (grayscale)
    or 3(rgb) channels. This is useful for plotting the weights of the first hidden
    layer of the network. If you want to plot the weights of later layer, where the
    number of input channels is arbitrary set use_color to false.
  
    If scale_each is true (default), the values in each filter will be scaled
    independently before plotting; this makes the features of each individual
    filter stand out more. If it's false, the scaling will be done globally
    across all filters; this allows one to compare the filters."""

    img = make_filter_image(layer, use_color=use_color, scale_each=scale_each)
    size = 4+max(img.size())/10
    fig, ax = plt.subplots(figsize=(size,size))
    imshow(img, ax)
