"""Create, train and test an RBM. Works in concert with fig_gen, fig_plot and loose_fun.

First create an RBM with certain numbers of hidden layer 1 (HL1) and hidden layer 2 (HL2) nodes.
Then train the network (unsupervised) using contrastive divergence (CD) learning, with method 'training1'.
Training data is obtained from file 'training1' or generated.
Subsequently add two classifiers using method 'addclassifiers', and train the weights to these classifiers using
'training2'.
Use 'reproductions' to visualize learning. This method prints the original and reproduction by the network of training
figures.

NOTE: this program assumes that the distance between areas is equal to the smallest area (usually equal to 32).
"""

import pickle
import random
import numpy as np
import fig_gen as fg  # Contains method for generating training and testing data
import loose_fun as lf  # Contains several methods, e.g. for activation spreading and plotting
import loose_fun_an as lfa # Contains several methods for performance and results analysis
from scipy import misc
from pylab import mean
import line_profiler
import matplotlib.pyplot as plt


class RBM(object):
    """Create, train and test an RBM with one visible layer, two hidden layers and one classifier layer.

    Class methods:
    training1 -- first, unsupervised, training round
    reproductions -- create reproductions of input data
    addclassifiers -- add classifiers nodes
    training2 -- second, supervised, training round
    testing -- test trained RBM
    """

    def __init__(self, name, nvis=900, nhid1=80, nhid2=400, nhid3=2, weights_stdev=0.01):
        """Define instance variables.

        :param name: RBM name, string, used in naming output files
        :param nvis: number of units (pixels) in visible layer (default: 900)
        :param nhid1: number of nodes in hidden layer 1 (default: 80)
        :param nhid2: number of nodes in hidden layer 2 (default: 400)
        :param nhid3: number of nodes in hidden layer 3 (classifiers) (default: 2)**
        :param weights_stdev: standard deviation in gaussian distribution of initial weights (default: 0.01)

        ** Note: methods 'addclassifiers' and 'training2' assume nhid3 = 2!
        """

        self.name = name  # Store RBM name
        self.nvis = nvis  # Set number of visible and hidden layer units
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.nhid3 = nhid3

        self.training_stage = {'layer': 0, 'epoch': 0, 'batch': 0}  # Variables to track object (training) stage
        self.trained1 = False
        self.added_classif = False
        self.trained2 = False

        # Some instance variables specified in other methods, placeholder value but if possible correct format
        self.tr1_learn_rate = 0.0
        self.tr2_learn_rate = 0.0

        self.classif1 = 0
        self.classif2 = 0

        # Initialize weight and bias weight matrices:
        self.weights_stdev = weights_stdev
        self.weights1 = weights_stdev * np.random.randn(nvis, nhid1)  # to small random values for the HLs
        self.weights2 = weights_stdev * np.random.randn(nhid1, nhid2)
        self.weights3 = weights_stdev * np.random.randn(nhid2, nhid3)
        # Old (alternative) methods, below methods recommended by Hinton
        # self.weightsvis_b = weights_stdev * np.random.randn(1, nvis)
        # self.weights1_b = weights_stdev * np.random.randn(1, nhid1)
        # self.weights2_b = weights_stdev * np.random.randn(1, nhid2)
        self.weights1_b = np.zeros((1, nhid1))  # to zeros for the bias weights
        self.weights2_b = np.zeros((1, nhid2))
        self.weights3_b = np.zeros((1, nhid3))
        self.weightsvis_b = np.zeros((1, 1))  # NOTE: will be set when data is obtained

    def training1(self, epochs=300, num_batches=320, learn_rate=0.01, load_data=True, pickle_rbm=True):
        """Train RBM unsupervised using CD-1 learning.

        Parameters:
        :param epochs: how many times the entire data set is fed to the network (default: 300)
        :param num_batches: the number of batches the training data is divided into (default: 320)
        :param learn_rate: learning rate used in CD learning (default: 0.01)
        :param load_data: how to obtain the training data, either load from file or generate (default: True (load))
        :param pickle_rbm: whether to store rbm as pickle file (default: True)

        Note: The data consists of an array, where each row corresponds to a training image.
        """

        # percentile_list = [5, 25, 50, 75, 95]
        # perclist_w1 = []
        # perclist_w2 = []
        # perclist_vb = []
        # perclist_h1b = []
        # perclist_h2b = []

        if self.trained1:
            print("This RBM already completed Training 1.")
        else:
            # Check if training 1 has been interrupted before, and get appropriate number of epochs.
            where_were_we = self.training_stage
            batch_advance = where_were_we['batch'] / num_batches
            zero_or_one = round(batch_advance)  # Note: number is approximate!
            epochs -= where_were_we['epoch'] - zero_or_one

            # Generate or load training data, depending on the get_data variable
            if load_data:
                data_dict = pickle.load(open("training1", "rb"))
            else:
                data_dict = fg.generatefigures()

            data = data_dict['data_array']

            # Initialize visible layer bias weights if training1 is just starting:
            if self.training_stage['epoch'] == 0 and self.training_stage['batch'] == 0:
                # Calculate probability that a certain pixel in training data is on:
                p_unit_on_arr = np.mean(data, axis=0)
                # Set visible layer bias weights to log(p/(1-p))
                self.weightsvis_b = np.log(p_unit_on_arr / (1 - p_unit_on_arr))
                self.weightsvis_b = self.weightsvis_b.reshape((1, 900))

            # Load data information to get the number of figures per batch (ond some other potentially useful nrs)
            num_fig_tot = data.shape[0]
            self.tr1_learn_rate = learn_rate

            """Determine number of figures per batch."""
            per_batch = num_fig_tot / num_batches
            if per_batch % 1 != 0:
                print('Warning Training 1: invalid number of figures per batch ({}).'.format(per_batch))
            else:
                """Training Layer 1"""
                per_batch = int(per_batch)

                for epoch in range(epochs):
                    # Shuffle data to divide randomly in mini-batches (note: variable 'data' is now scrambled)
                    np.random.shuffle(data)

                    for batch in range(num_batches):  # For each mini-batch
                        # Calculate indices belonging to this batch
                        this_batch = [(x + batch) for x in range(int(per_batch))]
                        # Store all figures (lines from data matrix) from this batch in a variable
                        data_thisbatch = data[this_batch, :]

                        # Create array with bias weights to add to calculated Vis and HL1 input
                        # NOTE: needs to be done per batch (rather than just once) to use updated weights!
                        vis_bias_arr = np.tile(self.weightsvis_b, (per_batch, 1))
                        hid1_bias_arr = np.tile(self.weights1_b, (per_batch, 1))

                        """Positive phase:
                        feed figure (visible layer) to the network, compute HL1 activation."""
                        pos_hid1_activprob, pos_hid1_activ = lf.act_forw(data_thisbatch, self.weights1, hid1_bias_arr)

                        """Negative phase:
                        calculate 'imagined' visible and HL1 activations."""
                        neg_vis_activprob = lf.act_back(pos_hid1_activ, self.weights1, vis_bias_arr,
                                                        only_activprob=True)
                        # NOTE: only use activprob instead of activ, faster. Common procedure, among others reduces
                        # sampling noise (see Hinton).
                        neg_hid1_activprob = lf.act_forw(neg_vis_activprob, self.weights1, hid1_bias_arr,
                                                         only_activprob=True)
                        # Note: no need to calculate neg_hid1_activ because it isn't used anywhere

                        """Update weights using the associations between Vis and HL1 activation.
                        NOTE: with 'association', the value (v_i * h_j) is meant, used in calculating
                        the weight change between nodes i and j in the visible and hidden layer, resp.
                        Template 3 uses the activation probabilities to calculate this, instead of the
                        actual (stochastic) activations. I will do the same, unclear in paper and think
                        it makes sense."""
                        batch_delta = lf.wupdate(data_thisbatch, pos_hid1_activprob, neg_vis_activprob,
                                                 neg_hid1_activprob, learn_rate)
                        # Update weight matrix:
                        self.weights1 += batch_delta

                        # Same procedure for the bias weights (given bias activation is always equal to 1):
                        vis_bias_delta, hl1_bias_delta = lf.wupdate_b(data_thisbatch, pos_hid1_activprob,
                                                                      neg_vis_activprob, neg_hid1_activprob, learn_rate)
                        self.weightsvis_b += vis_bias_delta
                        self.weights1_b += hl1_bias_delta

                        # Check relative size of weight updates:
                        # relative_updw1 = batch_delta / self.weights1
                        # perc_updw1 = np.percentile(relative_updw1, percentile_list)
                        # perclist_w1.append(perc_updw1)
                        #
                        # relative_updvb = vis_bias_delta / self.weightsvis_b
                        # perc_updvb = np.percentile(relative_updvb, percentile_list)
                        # perclist_vb.append(perc_updvb)
                        #
                        # relative_updh1b = hl1_bias_delta / self.weights1_b
                        # perc_updh1b = np.percentile(relative_updh1b, percentile_list)
                        # perclist_h1b.append(perc_updh1b)

                        self.training_stage = {'layer': 1, 'epoch': epoch, 'batch': batch}

                        # Print reproduction error per figure every now and then
                        if (batch + 1) % 10 == 0:
                            err_perfig = np.sum((data_thisbatch - neg_vis_activprob) ** 2) / per_batch
                            print('Training 1, HL1, epoch {}, batch {}, error {}'.format((epoch + 1), (batch + 1),
                                                                                         err_perfig))

                        # Save RBM as pickle
                        if (epoch + 1) % 10 == 0:
                            pickle.dump(self, open('{}_tr1_in_progress'.format(self.name), 'wb'))

                """Training Layer 2
                Note: HL2 is trained with the same dataset as HL1.
                For more extensive annotations: see Training Layer 1 above."""
                for epoch in range(epochs):
                    # Shuffle data again to obtain different mini-batches for every epoch
                    np.random.shuffle(data)

                    for batch in range(num_batches):
                        this_batch = [(x + batch) for x in range(int(per_batch))]
                        data_thisbatch = data[this_batch, :]

                        hid1_bias_arr = np.tile(self.weights1_b, (per_batch, 1))
                        hid2_bias_arr = np.tile(self.weights2_b, (per_batch, 1))

                        """Positive Phase:
                        feed input through trained weights1 to get HL1 input and activation,
                        on to compute HL2 input and activation."""
                        pos_hid1_activprob, pos_hid1_activ = lf.act_forw(data_thisbatch, self.weights1, hid1_bias_arr)
                        pos_hid2_activprob, pos_hid2_activ = lf.act_forw(pos_hid1_activ, self.weights2, hid2_bias_arr)

                        """Negative Phase:
                        calculate 'imagined' HL1 and HL2 activations."""
                        neg_hid1_activprob = lf.act_back(pos_hid2_activ, self.weights2, hid1_bias_arr,
                                                         only_activprob=True)
                        neg_hid2_activprob = lf.act_forw(neg_hid1_activprob, self.weights2, hid2_bias_arr,
                                                         only_activprob=True)

                        """Update weights using associations between HL1 and HL2 activations."""
                        batch_delta = lf.wupdate(pos_hid1_activprob, pos_hid2_activprob, neg_hid1_activprob,
                                                 neg_hid2_activprob, learn_rate)
                        self.weights2 += batch_delta

                        hl1_bias_delta, hl2_bias_delta = lf.wupdate_b(pos_hid1_activprob, pos_hid2_activprob,
                                                                      neg_hid1_activprob, neg_hid2_activprob,
                                                                      learn_rate)
                        self.weights1_b += hl1_bias_delta
                        self.weights2_b += hl2_bias_delta

                        # Check relative size of weight updates:
                        # relative_updw2 = batch_delta / self.weights2
                        # perc_updw2 = np.percentile(relative_updw2, percentile_list)
                        # perclist_w2.append(perc_updw2)
                        #
                        # relative_updh1b = hl1_bias_delta / self.weights1_b
                        # perc_updh1b = np.percentile(relative_updh1b, percentile_list)
                        # perclist_h1b.append(perc_updh1b)
                        #
                        # relative_updh2b = hl2_bias_delta / self.weights2_b
                        # perc_updh2b = np.percentile(relative_updh2b, percentile_list)
                        # perclist_h2b.append(perc_updh2b)

                        self.training_stage = {'layer': 2, 'epoch': epoch, 'batch': batch}

                        # Print reproduction error every now and then
                        if (batch + 1) % 10 == 0:
                            err_perfig = np.sum((pos_hid1_activprob - neg_hid1_activprob) ** 2) / per_batch
                            print('Training 1, HL2, epoch {}, batch {}, error {}'.format((epoch + 1), (batch + 1),
                                                                                         err_perfig))
                        if (epoch + 1) % 10 == 0:
                            pickle.dump(self, open('{}_tr1_in_progress'.format(self.name), 'wb'))

                self.trained1 = True  # Record training 1

                # Plot weight updates
                # lfa.percplot(perclist_w1, percentile_list, 'weight_updates_w1')
                # lfa.percplot(perclist_w2, percentile_list, 'weight_updates_w2')
                # lfa.percplot(perclist_vb, percentile_list, 'weight_updates_vb', sety=False)
                # lfa.percplot(perclist_h1b, percentile_list, 'weight_updates_h1')
                # lfa.percplot(perclist_h2b, percentile_list, 'weight_updates_h2')

                # Store RBM in output file for processing depending on variable pickle_rbm:
                if pickle_rbm:
                    pickle.dump(self, open('{}_tr1_completed'.format(self.name), 'wb'))

    def addclassifiers(self, classif1=8, classif2=16, pickle_rbm=True):
        """Add two classifiers (and corresponding weights) to the network.

        Parameters:
        :param classif1: first classifier (default: 8)
        :param classif2: second classifier (default: 16)
        :param pickle_rbm: whether to store rbm after this operation (default: True)
        """

        self.classif1 = classif1  # Store first comparison number
        self.classif2 = classif2  # Store second comparison number

        weights_stdev = 0.1  # Create corresponding weight matrix
        self.weights3 = weights_stdev * np.random.randn(self.nhid2, self.nhid3)

        self.added_classif = True  # Record adding classifiers

        if pickle_rbm:
            if self.trained1:
                pickle.dump(self, open('{}_tr1_complete_added_classif'.format(self.name), 'wb'))
            else:
                pickle.dump(self, open('{}_added_classif'.format(self.name), 'wb'))

    def reproductions(self, number=5, numerosity=5, area=128, dataname='training1', save_im=True, disp_im=False,
                      filename='reproduction'):
        """Produce reproductions of data figures.

        Parameters:
        :param number: desired number of reproductions (default: 5)
        :param numerosity: desired numerosity of objects in figures to be reproduced (default: 5)
        :param area: desired area of objects in figures to be reproduced (default: 128)
        :param dataname: name data file containing figures to be reproduced (default: 'training1')
        :param save_im: whether to save the plot as an image file (default: True)
        :param disp_im: whether to show the image on screen (default: False)
        (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
        :param filename: desired filename for the output files
        :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
        """

        if not self.trained1:
            print('Note: this RBM has not been trained yet. Reproductions are random.')

        data_dict = pickle.load(open(dataname, 'rb'))
        data = data_dict['data_array']
        data_info = data_dict['parameter_list']
        num_numerosities = data_info[1]
        per_area_numerosity = data_info[2]
        area_dis = data_info[3]
        areas = data_info[4]
        numerosities = data_info[5]

        # Check validity of input
        if area not in areas:
            print('Please enter an area from this list: {}.'.format(areas))
        elif numerosity not in numerosities:
            print('Please enter a numerosity from this list: {}.'.format(numerosities))
        elif number > per_area_numerosity:
            print('Please enter a number below or equal to {}.'.format(per_area_numerosity))
        else:
            """Determine which figures to reproduce"""
            # Get all valid figure-indices:
            area_index = area / area_dis

            # Original (for reference):
            # starting_point = ((area_index-1) * self.num_numerosities * self.per_area_numerosity) + \
            # ((numerosity-1) * self.per_area_numerosity)
            # des_range = [ (starting_point + x) for x in range(self.per_area_numerosity)]

            starting_point = ((area_index - 1) * num_numerosities * per_area_numerosity) + \
                             ((numerosity - 1) * per_area_numerosity)
            des_range = [(starting_point + x) for x in range(per_area_numerosity)]

            # Generate random indices within this range:
            random_indices = random.sample(des_range, number)
            # Turn elements to integers for compatibility:
            random_indices_int = list(map(int, random_indices))

            # Create array containing desired original images
            originals = data[random_indices_int, :]

            # Create bias weight-arrays of correct shape:
            vis_bias_arr = np.tile(self.weightsvis_b, (number, 1))
            hid1_bias_arr = np.tile(self.weights1_b, (number, 1))
            hid2_bias_arr = np.tile(self.weights2_b, (number, 1))

            """Forward phase (feed activation from visible layer to HL2)."""
            for_hid1_activprob, for_hid1_activ = lf.act_forw(originals, self.weights1, hid1_bias_arr)
            for_hid2_activprob, for_hid2_activ = lf.act_forw(for_hid1_activ, self.weights2, hid2_bias_arr)

            """Backward Phase (calculate 'imagined' HL1 and Vis activations)."""
            back_hid1_activprob, back_hid1_activ = lf.act_back(for_hid2_activprob, self.weights2, hid1_bias_arr)
            reconstructions = lf.act_back(back_hid1_activ, self.weights1, vis_bias_arr, only_activprob=True)

            """Plotting the probabilities as the reconstructions, other option is to plot the
            actual activity (use lines below)."""
            # back_vis_activ = back_vis_activprob > np.random.rand(number, self.nvis)
            # back_vis_activ.astype(int)
            # reconstructions = back_vis_activ

            # Loop to print originals and reconstructions
            for item in range(number):
                orig = originals[item, :]
                lf.basicplot(orig, file_name="{}_a{}n{}_{}_orig".format(filename, area, numerosity, (item + 1)),
                             disp_im=disp_im, save_im=save_im)
                recon = reconstructions[item, :]
                lf.basicplot(recon, file_name="{}_a{}n{}_{}_repr".format(filename, area, numerosity, (item + 1)),
                             disp_im=disp_im, save_im=save_im)

    def reproductions_other(self, input_image, filename, save_im=True, disp_im=False):
        """Produce reproductions of any image that is 30 x 30 pixels and already converted to 1x900 numpy array.

        Parameters:
        :param input_image: name data file containing figures to be reproduced
        :param filename: desired file name for reproduction image output file
        :param save_im: whether to save the plot as an image file (default: True)
        :param disp_im: whether to show the image on screen (default: False)
        (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
        :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
        """

        if not self.trained1:
            print('Note: this RBM has not been trained yet. Reproductions are random.')

        """Prepare image for RBM processing"""
        original_square_rgb = plt.imread('{}.png'.format(input_image))         # Load image
        original_square_inv_large = mean(original_square_rgb, 2)                # RGB --> Grayscale
        div_max = original_square_inv_large.max()                               # Determine highest value
        original_square_inv = original_square_inv_large / float(div_max)        # Divide by max to get values [0-1]
        original_square = 1 - original_square_inv                               # Invert image

        original = original_square.reshape((1, 900))                            # Reshape image

        """Forward phase (feed activation from visible layer to HL2)."""
        for_hid1_activprob, for_hid1_activ = lf.act_forw(original, self.weights1, self.weights1_b)
        for_hid2_activprob, for_hid2_activ = lf.act_forw(for_hid1_activ, self.weights2, self.weights2_b)

        """Backward Phase (calculate 'imagined' HL1 and Vis activations)."""
        back_hid1_activprob, back_hid1_activ = lf.act_back(for_hid2_activprob, self.weights2, self.weights1_b)
        reconstruction = lf.act_back(back_hid1_activ, self.weights1, self.weightsvis_b, only_activprob=True)

        # (don't know why this is necessary here but not with other reconstruction-function)
        reconstruction = reconstruction.reshape((30, 30))

        """Plotting the probabilities as the reconstructions, other option is to plot the
        actual activity (use lines below)."""
        # back_vis_activ = back_vis_activprob > np.random.rand(number, self.nvis)
        # back_vis_activ.astype(int)
        # reconstructions = back_vis_activ

        # Print original and reconstruction
        lf.basicplot(reconstruction, file_name="{}_repr".format(filename),
                     disp_im=disp_im, save_im=save_im)

    def training2(self, epochs=300, num_batches=320, learn_rate=0.1, load_data=True, pickle_rbm=True):
        """Train the classifier weights using supervised learning and the delta rule.

        Parameters:
        :param epochs: how many times the entire data set is fed to the network (default: 300)
        :param num_batches: the number of batches the training data is divided into (default: 320)
        :param learn_rate: learning rate used in perceptron learning (default: 0.1)
        :param load_data: whether to load the training data from file or generate it (default: True (load))
        :param pickle_rbm: whether to pickle rbm after this operation (default: True)
        """

        if self.trained2:
            print("This RBM already completed Training 2.")
        elif not self.trained1:
            print("Please complete Training 1 with this RBM before starting Training 2.")
        elif not self.added_classif:
            print("Please add classifiers to this RBM before starting Training 2.")
        else:
            classif_list = [self.classif1, self.classif2]
            for index in range(len(classif_list)):
                """Train Classifiers (classifiers are trained separately to offer balanced training data)"""
                """Get training data, from pickle or dictionary"""
                if load_data:
                    data_dict = pickle.load(open('tr2_{}'.format(classif_list[index]), 'rb'))
                else:
                    data_dict = fg.generatefigures(data_stage='tr2_{}'.format(classif_list[index]), num_areas=8,
                                                   area_dis=32, num_numerosities=((2 * classif_list[index]) - 1),
                                                   pickle_data=False)

                data = data_dict['data_array']
                data_array_info = data_dict['data_array_info']

                self.tr2_learn_rate = learn_rate

                corr_activ = lf.corr_ans_hid3(data_array_info, data.shape[0], classif_list[index])[0]
                # Note: only store first output of function by using function()[0]

                """Determine number of figures per batch."""
                per_batch = data.shape[0] / num_batches

                if per_batch % 1 != 0:
                    print('Warning Training 2: invalid number of figures per batch ({}).'.format(per_batch))
                else:
                    """Training Layer 3"""
                    per_batch = int(per_batch)

                    # Create array with bias weights to add to HL1 and HL2 input (only needs to be done once because
                    # weights are fixed)
                    hid1_bias_arr = np.tile(self.weights1_b, (per_batch, 1))
                    hid2_bias_arr = np.tile(self.weights2_b, (per_batch, 1))

                    # While-loop break
                    finish_tr = False
                    epoch = 0

                    while not finish_tr:                                # While loop counting epochs
                        # Shuffle data and correct answers to divide randomly in mini-batches
                        data, corr_activ = lf.sync_shuffle(data, corr_activ)

                        for batch in range(num_batches):  # For each mini-batch
                            # Calculate indices belonging to this batch
                            this_batch = [(x + batch) for x in range(int(per_batch))]
                            # Store all figures (lines from data matrix) from this batch in a variable
                            data_thisbatch = data[this_batch, :]
                            corr_activ_thisbatch = corr_activ[this_batch, :]

                            # Create array with bias weights to add to HL3 input
                            # NOTE: needs to be done per batch (rather than just once) to use updated weights
                            hid3_bias_arr = np.tile(self.weights3_b, (per_batch, 1))

                            # Propagate activation from the visible layer to HL3
                            hid1_activprob, hid1_activ = lf.act_forw(data_thisbatch, self.weights1, hid1_bias_arr)
                            hid2_activprob, hid2_activ = lf.act_forw(hid1_activ, self.weights2, hid2_bias_arr)
                            hid3_activ = lf.act_forw_hid3(hid2_activ, self.weights3, hid3_bias_arr)

                            # Calculate error and update weights
                            hid3_wupdate, hid3_wupdate_b = lf.wupdate_hid3(hid3_activ[:, index], corr_activ_thisbatch,
                                                                           hid2_activ, learn_rate)


                            # Add weight update to weight arrays
                            self.weights3[:, index] += hid3_wupdate.reshape((self.nhid2, 1))
                            self.weights3_b[0, index] += hid3_wupdate_b

                            # print('hid3_wupdate[:10] = {}'.format(hid3_wupdate.reshape(self.nhid2)[:10]))
                            # print('weights3[:10, index] = {}'.format(self.weights3[:10, index]))
                            # print('hid3_wupdate_b = {}'.format(hid3_wupdate_b))
                            # print('weights3_b[0, index] = {}'.format(self.weights3_b[0, index]))

                            self.training_stage = {'layer': 3, 'epoch': epoch, 'batch': batch}

                            # Calculate sum squared error (see Rosenblatt, 1958)
                            err_sum = .5 * (corr_activ_thisbatch - hid3_activ) ** 2
                            sumsq_err = np.sum(err_sum)

                            if sumsq_err == 0:
                                print('Finish, classif {} of {}, epoch {}, batch {}'.format((index + 1),
                                                                                            len(classif_list),
                                                                                            (epoch + 1), (batch + 1),
                                                                                            sumsq_err))
                                finish_tr = True
                                break

                            # Print sum of squares every now and then
                            if (batch + 1) % 10 == 0:
                                print('Tr 2, classif {} of {}, epoch {}, batch {}, error {}'.format((index + 1),
                                                                                                    len(classif_list),
                                                                                                    (epoch + 1),
                                                                                                    (batch + 1),
                                                                                                    sumsq_err))

                            # Save RBM as pickle
                            if (epoch + 1) % 10 == 0:
                                pickle.dump(self, open('{}_tr2_in_progress'.format(self.name), 'wb'))

                        epoch += 1
                        if epoch >= epochs:
                            break

            self.trained2 = True  # Record result of training 2

            if pickle_rbm:
                pickle.dump(self, open('{}_tr2_completed'.format(self.name), 'wb'))

    def testing(self, pickle_name='testing_results', load_data=True, pickle_rbm=True):
        """Test the trained network, record the results.

        Parameters:
        :param pickle_name: name of the output file (default: 'testing_results')
        :param load_data: load testing data from file or generate it (default: True (load))
        :param pickle_rbm: whether to pickle rbm after this operation (default: True)
        :return: file or dictionary with outcome and supplementary information, depending on variable pickle_data.
         Dictionary elements, all arrays of the same shape (number of test figures, 2):
          'hid3_activ': array of actual HL3 activity in response to testing data
          'classif_correct': array of desired HL3 activity in response to testing data
          'performance': array with same shape as hid3_activ, with 1 when activity was correct
          'data_num': array with numerosity of testing figures (all rows being identical: (numerosity, numerosity))
          'classif_arr': array with classifiers values (all rows being identical: (classif1, classif2))
          'ratios': array with the ratio of the data numerosity to the classifier numerosities
        """

        # Check if network is ready for testing:
        if not self.trained1:
            print("Please complete Training 1 before testing this RBM.")
        elif not self.added_classif:
            print("Please add classifiers before testing this RBM.")
        elif not self.trained2:
            print("Please complete Training 2 before testing this RBM.")
        else:
            """Get test data, depending on variable get_data."""
            if load_data:
                data_dict = pickle.load(open('testing', 'rb'))
            else:
                data_dict = fg.generatefigures()

            data = data_dict['data_array']
            data_array_info = data_dict['data_array_info']
            test_num = data.shape[0]

            classif_correct, data_num, classif_arr = lf.corr_ans_hid3(data_array_info, test_num, self.classif1,
                                                                      self.classif2)

            """Create bias arrays to add to inputs (see 'training1' for annotation)."""
            hid1_bias_arr = np.tile(self.weights1_b, (test_num, 1))
            hid2_bias_arr = np.tile(self.weights2_b, (test_num, 1))
            hid3_bias_arr = np.tile(self.weights3_b, (test_num, 1))

            """Feed activation from visible layer to HL3 to get classifier values."""
            hid1_activprob, hid1_activ = lf.act_forw(data, self.weights1, hid1_bias_arr)
            hid2_activprob, hid2_activ = lf.act_forw(hid1_activ, self.weights2, hid2_bias_arr)
            hid3_activ = lf.act_forw_hid3(hid2_activ, self.weights3, hid3_bias_arr)

            """Record and return performance, also store numerical ratio."""
            # Determine whether obtained classifications are correct and store in 'performance' array[test_num, 2]
            performance = np.equal(hid3_activ, classif_correct)
            ratios = data_num / classif_arr

            return_dict = {'hid3_activ': hid3_activ, 'classif_correct': classif_correct, 'performance': performance,
                           'data_num': data_num, 'classif_arr': classif_arr, 'ratios': ratios}

            if pickle_rbm:
                pickle.dump(return_dict, open('{}_{}'.format(self.name, pickle_name), 'wb'))
            else:
                return return_dict


test_epochs = 3

if __name__ == "__main__":
    my_rbm = RBM(name="test_rbm")
    my_rbm.training1(epochs=test_epochs)
    # my_rbm.reproductions()
    # my_rbm.addclassifiers()
    # my_rbm.testing(pickle_name='test_run1')
    # my_rbm.training2(epochs=test_epochs)
    # my_rbm.testing(pickle_name='test_run2')

# Abel:
# Tweede model: ik ben cool (ibc)
# Derde model: lnd (lodewijk en denzel)
