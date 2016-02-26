"""Code contains some loose functions used in RBM class (RBM.py) and plotting.

Functions:

sigmoid(arr1)
duplicate_and_vstack(row_arr, num_rows)
act_forw(data_vis, weight_arr, hid_bias_arr, only_activprob=False)
act_forw_hid3(hid2_activ, weight_arr, bias_arr, theta=0)
act_back(data_hid, weight_arr, vis_bias_arr, only_activprob=False)
wupdate(data_vis, pos_hid_activprob, neg_vis_activprob, neg_hid_activprob, learn_rate)
wupdate_b(data_vis, pos_hid_activprob, neg_vis_activprob, neg_hid_activprob, learn_rate)
wupdate_hid3(hid3_activ, corr_activ, hid2_activ, learn_rate)
sync_shuffle(arr_a, arr_b)
corr_ans_hid3(fig_info_list, test_num, classif1, classif2)

basicplot(fig_array, file_name, file_extension=".png", save_fig=True, show_fig=False)
basiccontourplot(w1_array, file_name, file_extension=".png", save_fig=True, show_fig=False)
"""


import math as mt
import numpy as np
import copy
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""Some functions mostly used in the BRM class."""


def sigmoid(arr1):
    """Logistic sigmoid function to transform input, used in the RBM class defined in RBM_manual.

    :param arr1: elements of this object to be converted
    :return: object of same type and shape as input (some_object)

    Note: Input is converted to a number in the range [0, 1]. A net-input of 0 returns .5.
    Approaches zero and one around -4 and 4, respectively. Output is same type and shape as input.
    """

    arr3 = (1 + np.exp(-arr1)) ** (-1)                          # sigmoid body
    return arr3                                                 # return answer array


def duplicate_and_vstack(row_arr, num_rows):
    """Create array of num_rows rows by duplicating and stacking row_vector.
    Among others useful for creating bias array to add to regular input, and creating reference arrays for classifiers.

    :param row_arr: array to be duplicated, shaped like row vector
    :param num_rows: number (int) of rows of input array (e.g. figures per batch)
    :return: array with row_arr as rows and num_rows rows, shape: (row_arr.shape[1], num_rows)
    """

    stacked_arr = copy.copy(row_arr)                            # Create first row of stacked array

    for _ in range(num_rows - 1):                               # Add new rows to the stacked array
        stacked_arr = np.vstack((stacked_arr, row_arr))

    return stacked_arr


def act_forw(data_vis, weight_arr, hid_bias_arr, only_activprob=False):
    """Calculate HL activation probability and activation given Vis activation, weights Vis-HL, and bias weights.

    :param data_vis: data to be forwarded (or: visible layer activation)
    :param weight_arr: weight array between Vis and HL
    :param hid_bias_arr: array of HL bias weights, with same number of rows as data array
    :param only_activprob: whether only the activprob, or also activity is needed (longer calculation) (default: False)
    :return: array of hidden layer activity probability and activity, resp. Shape: (per_batch, nhid).
    Hidden layer activity will NOT be calculated or returned when only_activprob is set to True
    """

    # Compute HL input by multiplying the visible layer activation with the weight matrix:
    hid_input = np.dot(data_vis, weight_arr)
    hid_input += hid_bias_arr                    # Add bias
    # Compute HL activation probabilities using sigmoid function defined above
    hid_activprob = sigmoid(hid_input)

    if only_activprob:
        return hid_activprob
    else:
        # To get the actual activations from the probabilities, compare probability with random number [0-1]:
        hid_activ = hid_activprob > np.random.rand(hid_bias_arr.shape[0], hid_bias_arr.shape[1])
        # convert boolean activation array into integer zeros and ones:
        hid_activ.astype(int)

        return hid_activprob, hid_activ                 # Return result


def act_forw_hid3(hid2_activ, weight_arr, bias_arr, theta=0):
    """Calculate HL3 activation given HL2 activation, weight and bias arrays, and input activation threshold.

    :param hid2_activ: HL2 activation array
    :param weight_arr: weight array between HL2 and HL3
    :param bias_arr: array of HL3 bias weights, with same number of rows as HL2 activation array
    :param theta: input threshold for unit activation (default: 0)
    :return: array of HL3 activity, shape: (per_batch, nhid3)
    """

    hid3_input = np.dot(hid2_activ, weight_arr)         # Calculate HL3 input
    hid3_input += bias_arr                              # Add bias

    hid3_activ = hid3_input > theta                     # Activity is 1 when input exceeds threshold
    hid3_activ.astype(int)                              # Convert boolean to integer array

    return hid3_activ                                   # Return result


def act_back(data_hid, weight_arr, vis_bias_arr, only_activprob=False):
    """Calculate Vis activation probability and activation given Vis activation, weights Vis-HL, and bias weights.

    :param data_hid: HL activation (/probability) vector
    :param weight_arr: weight array between Vis and HL
    :param vis_bias_arr: array of Vis bias weights with same number of rows as data array
    :param only_activprob: whether only the activprob, or also activity is needed (longer calculation) (default: False)
    :return: array of vis-layer activity probability and activity, resp. Shape: (per_batch, nvis).
    Vis_activ will NOT be calculated or returned when only_activprob is set to True
    """

    # Compute Vis input by multiplying HL activation with the transpose of the weight matrix:
    vis_input = np.dot(data_hid, np.transpose(weight_arr))
    vis_input += vis_bias_arr                   # Add bias
    # Compute Vis activation probabilities using sigmoid function defined above
    vis_activprob = sigmoid(vis_input)

    if only_activprob:
        return vis_activprob
    else:
        # To get the actual activations from the probabilities, compare probability with random number [0-1]:
        vis_activ = vis_activprob > np.random.rand(vis_bias_arr.shape[0], vis_bias_arr.shape[1])
        # Just to be sure, convert boolean activation array into integer zeros and ones:
        vis_activ.astype(int)
        return vis_activprob, vis_activ


def wupdate(data_vis, pos_hid_activprob, neg_vis_activprob, neg_hid_activprob, learn_rate):
    """Calculate update for regular weight matrices during training 1.

    :param data_vis: visible layer input data (if available, activprob rather than activ) (positive phase)
    :param pos_hid_activprob: hidden layer activprob (positive phase)
    :param neg_vis_activprob: visible layer activprob (negative phase)
    :param neg_hid_activprob: hidden layer activprob (negative phase)
    :param learn_rate: learning rate
    :return: array of weights updates. Shape: (nvis, nhid)
    """

    # Calculate the positive and negative associations:
    pos_assoc = np.dot(np.transpose(data_vis), pos_hid_activprob)
    neg_assoc = np.dot(np.transpose(neg_vis_activprob), neg_hid_activprob)

    # Compute difference between input-output (error), adjusted for batch-size:
    batch_delta = learn_rate * ((pos_assoc - neg_assoc) / data_vis.shape[0])
    # Divide by batch size to allow learning rate to be constant at different batch sizes (see Hinton)

    return batch_delta


def wupdate_b(data_vis, pos_hid_activprob, neg_vis_activprob, neg_hid_activprob, learn_rate):
    """Calculate update for bias weight matrices during training 1.

    :param data_vis: visible layer input data (if available, activprob rather than activ) (positive phase)
    :param pos_hid_activprob: hidden layer activprob (positive phase)
    :param neg_vis_activprob: visible layer activprob (negative phase)
    :param neg_hid_activprob: hidden layer activprob (negative phase)
    :param learn_rate: learning rate
    :return: two arrays of weight updates for bias weights for vis and hid layer. Shape: (1, nvis) and (1, nhid), resp.
    """

    per_batch = data_vis.shape[0]

    # Similar procedure as for normal weights, given bias activation is always equal to 1:
    pos_assoc_vis_b = np.dot(np.ones((1, per_batch)), data_vis)
    neg_assoc_vis_b = np.dot(np.ones((1, per_batch)), neg_vis_activprob)
    vis_bias_delta = learn_rate * ((pos_assoc_vis_b - neg_assoc_vis_b) / per_batch)
    # Divide by batch size to allow learning rate to be constant at different batch sizes (see Hinton)

    pos_assoc_hid_b = np.dot(np.ones((1, per_batch)), pos_hid_activprob)
    neg_assoc_hid_b = np.dot(np.ones((1, per_batch)), neg_hid_activprob)
    hid_bias_delta = learn_rate * ((pos_assoc_hid_b - neg_assoc_hid_b) / per_batch)

    return vis_bias_delta, hid_bias_delta


def wupdate_hid3(hid3_activ, corr_activ, hid2_activ, learn_rate):
    """Calculate weight updates for HL2 - HL3 weight matrix and HL3 bias weights during training 2 using the delta rule.

    :param hid3_activ: actual HL3 (classifier) activation array
    :param corr_activ: desired classifier activation array
    :param hid2_activ: HL2 activation array
    :param learn_rate: learning rate
    :return: two arrays of weights and bias weights updates. Shape: (nhid2, nhid3) and (1, nhid3), resp.
    """

    per_batch = hid3_activ.shape[0]
    delta = corr_activ - hid3_activ                                 # Array[per_batch, 1]
    delta_times_hid2activ = np.dot(np.transpose(hid2_activ), delta)   # Array[nhid2, 1]

    update_w = (learn_rate * delta_times_hid2activ) / per_batch     # Array[nhid2, nhid3]
    # Divide by batch size to allow learning rate to be constant at different batch sizes (see Hinton)
    update_w_b = learn_rate * np.mean(delta, axis=0)                # Array[1, nhid3]

    return update_w, update_w_b


def sync_shuffle(arr_a, arr_b):
    """Shuffle arrays arr_a and arr_b along their first dimension such that corresponding rows remain so in output.

    :param arr_a: first array
    :param arr_b: second array
    :return: shuffled arrays arr_a and arr_b. Shape: arr_a.shape() and arr_b.shape(), resp.
    """

    nrows_a = arr_a.shape[0]
    nrows_b = arr_b.shape[0]

    if not nrows_a == nrows_b:
        print("Please enter two arrays with corresponding numbers of rows.")
    else:
        shuffled_indices = np.random.permutation(nrows_a)
        return arr_a[shuffled_indices], arr_b[shuffled_indices]


def corr_ans_hid3(fig_info_list, test_num, classif):
    """Create an array with the desired HL3 output for a given dataset.

    :param fig_info_list: list of lists containing numerosity of figures (format: data_array_info in fig_gen.py)
    :param test_num: number of figures in the test data set
    :param classif: value of the classifier
    :return: array of correct classifier values, array of the numerosities of the test figures, and array of the
    classifier values (shape of all: (test_num, nhid3))
    """

    # Create array (test_num x 1) with the figure numerosities:
    data_num = np.zeros((test_num, 1))
    data_num[:, 0] = [fig_info_list[x][2] for x in range(test_num)]

    # Create array (test_num x 1) with the classifier numerosity:
    classif_row = np.array([classif])
    classif_arr = duplicate_and_vstack(classif_row, test_num)

    # Create boolean array with correct classifier values:
    classif_correct = data_num > classif_arr
    classif_correct.astype(int)

    return classif_correct, data_num, classif_arr


"""Some functions for plotting"""


def basicplot(fig_array, file_name, file_extension=".png", save_im=True, disp_im=False):
    """Basic plot function for vectors or arrays (fig_array), used for plotting in fig_plot_info.

    Parameters:
    :param fig_array: array containing figures to be plotted
    :param file_name: desired output file name, string
    :param file_extension: desired output file extension (default: ".png")
    :param save_im: whether to save the plot as an image file (default: True)
    :param disp_im: whether to display the image on screen (default: False)
    (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
    :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
    """
    
    # Check if figure is still a vector. If so, reshape into square array.
    if len(fig_array.shape) == 1:
        side = mt.sqrt(len(fig_array))
        new_array = np.reshape(fig_array, (side, side))
    else:
        new_array = fig_array

    fig = plt.figure()

    plt.imshow(new_array, interpolation='none')         # Make image handle, remove blurring
    image_axes = plt.axes()                             # Make axes handle
    image_axes.axes.get_yaxis().set_visible(False)      # Disable y-axis
    image_axes.axes.get_xaxis().set_visible(False)      # Disable x-axis

    if disp_im:
        plt.show()                                          # To show the plot on screen in python

    if save_im:
        # Save to (.png) file, remove white border
        fig.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

    plt.close(fig)  # Clear current figure


def basiccontourplot(w1_array, file_name, file_extension=".png", save_im=True, disp_im=False):
    """Contour plot function for trained RBM [VIS->HID1] weights.

    Parameters:
    :param w1_array: array of weights to be plotted
    :param file_name: desired output file name, string
    :param file_extension: desired output file extension (default: ".png")
    :param save_im: whether to save the plot as an image file (default: True)
    :param disp_im: whether to show the image on screen (default: False)
    (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
    :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
    """

    # Check if weights are still shaped as a vector. If so, reshape into square array.
    if len(w1_array.shape) == 1:
        side = mt.sqrt(len(w1_array))
        z = np.reshape(w1_array, (side, side))
    else:
        z = w1_array

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    x, y = np.mgrid[:z.shape[0], :z.shape[1]]

    # X, Y, Z = axes3d.get_test_data(w1_array)

    # ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.3)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet)

    # cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    # cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    # cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

    # ax.set_xlabel('X')
    # ax.set_xlim(-40, 40)
    # ax.set_ylabel('Y')
    # ax.set_ylim(-40, 40)
    # ax.set_zlabel('Z')
    # ax.set_zlim(-100, 100)

    if disp_im:
        plt.show()

    if save_im:
        # Save to (.png) file, remove white border
        fig.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

    plt.close(fig)  # Clear current figure
