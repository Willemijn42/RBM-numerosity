"""Code containing some functions to analyze and plot the performance of RBM.

Functions:

"""

import numpy as np
import matplotlib.pyplot as plt


# sigmoid result line plot (with different surf areas (also different num?))
# sigmoid result scatter plot, including fitted line
# determine rbm Weber fraction (include in sigmoid plot?)


default_x = "Numerical ratio (log scale)"
default_y = "Percentage (/response 'larger')"
numerosities = np.array([(x + 1) for x in range(32)])


def lineplotperarea(arr1, arr2, arr3, arr4, arr5, lab1="64 px", lab2="96 px", lab3="128 px", lab4="160 px",
                    lab5="192 px", ratio_low=0.7, ratio_high=1.4, x_label=default_x, y_label=default_y, save_im=True,
                    disp_im=False, file_name="lineplotperarea", file_extension=".png"):
    # Reproduce paper plot

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(numerosities, arr1, 'b', label=lab1)
    ax.plot(numerosities, arr2, 'g', label=lab2)
    ax.plot(numerosities, arr3, 'r', label=lab3)
    ax.plot(numerosities, arr4, 'c', label=lab4)
    ax.plot(numerosities, arr5, 'purple', label=lab5)

    ax.legend()

    if disp_im:
        fig.show()

    if save_im:
        # Save to (.png) file, remove white border
        plt.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

    plt.close(fig)  # Clear current figure


def lineplotperarearest(arr1, arr2, arr3, arr4, lab1="32 px", lab2="128 px", lab3="224 px", lab4="256 px",
                        ratio_low=0.7, ratio_high=1.4, x_label=default_x, y_label=default_y, save_im=True,
                        disp_im=False, file_name="lineplotperarearest", file_extension=".png"):
    # reproduce paper plot additional info

    if disp_im:
        plt.show()

    if save_im:
        # Save to (.png) file, remove white border
        plt.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

    plt.close(fig)  # Clear current figure
    pass


def lineplotaltdata(arr1, lab1, arr2, lab2, arr3, lab3, arr4, lab4, arr5, lab5, ratio_low=0.7, ratio_high=1.4,
                    x_label=default_x, y_label=default_y, save_im=True, disp_im=False, file_name="lineplotperarea",
                    file_extension=".png"):
    # reproduce paper plot alternative training sets

    if disp_im:
        plt.show()

    if save_im:
        # Save to (.png) file, remove white border
        plt.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

    plt.close(fig)  # Clear current figure

    pass

def scatterplotregrcoeff(area_arr, num_arr, x_label="ß cumulative area", y_label="ß numerosity", save_im=True,
                         disp_im=False, file_name="lineplotperarea", file_extension=".png"):
    # reproduce paper plot with regression coefficients wrp area and numerosity

    if disp_im:
        plt.show()

    if save_im:
        # Save to (.png) file, remove white border
        plt.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

    plt.close(fig)  # Clear current figure

    pass


def percplot(data, perclist, file_name='weight_update_percentile', file_extension='.png', save_im=True, disp_im=False,
             sety=True):
    """Plot percentiles of weight update information during 'training1' of RBM object (RBM_m.py)

    :param data: list of lists, each sublist containing the same percentiles of the weight update matrices
    :param perclist: list of used percentiles
    :param file_name: name of optional output file (default: 'weight_update_percentile')
    :param file_extension: extension of optional output file (default: '.png')
    :param save_im: whether to save image (default: True)
    :param disp_im: whether to show image (default: False)
    :param sety: whether to set the y-range to predefined limits (default: True)
    :return: nothing, displays plot or saves plot to output file
    """

    fig = plt.figure()
    lines = plt.plot(data)
    plt.ylabel('update / weight')
    plt.xlabel('batch number')

    if sety:
        plt.ylim([-0.1, 0.1])

    plt.figlegend(lines, perclist, 'upper right')

    if disp_im:
        plt.show()

    if save_im:
        # Save to (.png) file, remove white border
        plt.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

    plt.close(fig)                  # Clear current figure
