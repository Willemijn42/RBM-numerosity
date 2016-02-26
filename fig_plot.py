"""
Define a class used to plot specific figures from (large) data set used for training and testing the model.
The plotting enables visual inspection and images can be used in reports or presentations.
Works with output from fig_generate.py.
The array is retrieved from a pickle file or dictionary, depending on the use of the script.
Then several different plotting- and information functions are defined.

Figures are stored as array lines that are 900 pixels long, representing a 30 x 30 pixel figure,
but different dimensions are allowed as long as the figures are square.
'obj_info' is a list of arrays, one array for each figure. Those arrays contain the info of all objects in the figure.
That means they contain the number of objects in the figure, times 4: the width, height, x- and y-coordinate, resp.
These last four parameters are contained in one line of the array.
E.g., to access the x-coordinate of the second object of the figure with index 200 (if it exists), use:
obj_info[200][1,2].

NOTE: this program assumes that the distance between areas is equal to the smallest area
(most likely equal to 32).

At the end of the program the functions are called and the desired parameters can be set.
"""


import numpy as np
import math as mt
import pickle
import random
import loose_fun as lf


class FigurePlotter(object):
    """Store general information for plotting the images.
    Functions for this class defined below deal with the plotting itself.
    """

    def __init__(self, dataset, dataset_name='tr1'):
        """Load the array containing the figures, get their length and shape.

        Parameters:
        :param dataset: dictionary containing figures and information
        :param dataset_name: string, name of dataset (will be used in naming output plot files) (default: 'tr1')
        """

        self.data_array = dataset['data_array']                           # Get array containing figures as vectors
        self.dataset_name = dataset_name

        self.parameter_list = dataset['parameter_list']                 # Get data parameters
        self.num_areas = self.parameter_list[0]
        self.num_numerosities = self.parameter_list[1]
        self.per_area_numerosity = self.parameter_list[2]
        self.dis_area = self.parameter_list[3]
        self.area_list = self.parameter_list[4]
        self.surf_area = dataset['surf_area']
        
        self.shape_array = self.data_array.shape                         # Get the shape of the array
        self.num_figures = self.shape_array[0]                          # Get the number of figures in the array
        self.len_vector = self.shape_array[1]                           # Get the length of the vectors
        self.dim_figure = mt.sqrt(self.len_vector)                      # Get the dimensions of the figures

        '''Handy lists and variables'''
        self.per_area = self.num_figures / self.num_areas
        self.area_list = [((x+1) * self.dis_area) for x in range(self.num_areas)]
        self.numerosities_list = [(x+1) for x in range(self.num_numerosities)]
        self.area_numerosity_list = [(x+1) for x in range(self.per_area_numerosity)]
        self.figures_list = [(x+1) for x in range(self.num_figures)]

    def plotrandomfigures(self, this_many, save_im=True, disp_im=False):
        """Plot some number of random figures.

        Parameters:
        :param this_many: number of images to plot
        :param save_im: whether to save the plot as an image file (default: True)
        :param disp_im: whether to show the image on screen (default: False)
        (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
        :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
        """
        
        if this_many not in self.figures_list:
            print("Please enter a number in the range 0 - {}.".format(self.num_figures))
        else:
            # Generate random number in the correct range:
            random_indices = random.sample(range(self.num_figures), this_many)
            
            for index in random_indices:
                # Get figure vector and plot figure
                fig_vector = self.data_array[index, :]
                file_name = "{}_{}_random".format(self.dataset_name, (index + 1))
                lf.basicplot(fig_vector, save_im=save_im, disp_im=disp_im, file_name=file_name)

    def plotareafigures(self, this_many, area, save_im=True, disp_im=False):
        """Plot some figures with fixed cumulative surface area.

        Parameters:
        :param this_many: number of images to plot
        :param area: cumulative surface area of the figures to be plotted
        :param save_im: whether to save the plot as an image file (default: True)
        :param disp_im: whether to show the image on screen (default: False)
        (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
        :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
        """
        
        if this_many not in self.figures_list:
            print("Please enter a number in the range 0 - {}.".format(self.num_figures))
        elif area not in self.area_list:
            print("Please enter a valid area from this list: {}.".format(self.area_list))
        else:
            area_index = area / self.dis_area
            starting_point = (area_index-1) * self.per_area
            des_range = [(starting_point + x) for x in range(self.per_area)]
            
            # Generate random indices within this area:
            random_indices = random.sample(des_range, this_many)
                
            for index in random_indices:
                # Get figure vector and plot figure
                fig_vector = self.data_array[index, :]
                file_name = "{}_{}_area_{}".format(self.dataset_name, (index + 1), area)
                lf.basicplot(fig_vector, save_im=save_im, disp_im=disp_im, file_name=file_name)

    def plotnumerosityfigures(self, this_many, numerosity, save_im=True, disp_im=False):
        """Plot some figures with fixed numerosity.

        Parameters:
        :param this_many: number of figures to be plotted
        :param numerosity: numerosity of the objects in the figures to be plotted
        :param save_im: whether to save the plot as an image file (default: True)
        :param disp_im: whether to show the image on screen (default: False)
        (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
        :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
        """
        
        if this_many not in self.figures_list:
            print("Please enter a number in the range 0 - {}.".format(self.num_figures))
        elif numerosity not in self.numerosities_list:
            print("Please enter a valid numerosity in the range 1 - {}.".format(self.num_numerosities))
        else:
            extended_ranges = []                        # To store the complete desired (interrupted)indices range
            
            for index in range(self.num_areas):
                
                starting_point = (index * self.num_numerosities * self.per_area_numerosity) + \
                                 ((numerosity-1) * self.per_area_numerosity)
                part_range = [(starting_point + x) for x in range(self.per_area_numerosity)]
                
                extended_ranges.extend(part_range)
            
            # Generate random indices within this area:
            random_indices = random.sample(extended_ranges, this_many)
                
            for index in random_indices:
                # Get figure vector and plot figure
                fig_vector = self.data_array[index, :]
                file_name = "{}_{}_numerosity_{}".format(self.dataset_name, (index + 1), numerosity)
                lf.basicplot(fig_vector, save_im=save_im, disp_im=disp_im, file_name=file_name)

    def plotcertainfigures(self, this_many, area, numerosity, save_im=True, disp_im=False):
        """Plot some figures with fixed area and numerosity.

        Parameters:
        :param this_many: number of figures to be plotted
        :param area: cumulative surface area of the figures to be plotted
        :param numerosity: numerosity of the objects in the figures to be plotted
        :param save_im: whether to save the plot as an image file (default: True)
        :param disp_im: whether to show the image on screen (default: False)
        (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
        :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
        """
        
        if this_many not in self.figures_list:
            print("Please enter a number in the range 0 - {}.".format(self.num_figures))
        elif area not in self.area_list:
            print("Please enter a valid area from this list: {}.".format(self.area_list))
        elif numerosity not in self.numerosities_list:
            print("Please enter a valid numerosity in the range 1 - {}.".format(self.num_numerosities))
        else:
            area_index = area / self.dis_area
            
            starting_point = ((area_index-1) * self.num_numerosities * self.per_area_numerosity) + \
                             ((numerosity-1) * self.per_area_numerosity)
            des_range = [(starting_point + x) for x in range(self.per_area_numerosity)]
            
            # Generate random indices within this area:
            random_indices = random.sample(des_range, this_many)
                
            for index in random_indices:
                # Get figure vector and plot figure
                fig_vector = self.data_array[index, :]
                file_name = "{}_{}_area_{}_numerosity_{}".format(self.dataset_name, (index + 1), area, numerosity)
                lf.basicplot(fig_vector, save_im=save_im, disp_im=disp_im, file_name=file_name)

    def plotthesefigures(self, listof_indices, save_im=True, disp_im=False):
        """Plot some specific figures using their indices.

        Parameters:
        :param listof_indices: list of the indices of the figures to be plotted
        :param save_im: whether to save the plot as an image file (default: True)
        :param disp_im: whether to show the image on screen (default: False)
        (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
        :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
        :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
        """
        
        print("Printing figures with indices:")
        print(listof_indices)
        
        for index in listof_indices:
            # Get figure vector and plot figure
            fig_vector = self.data_array[index, :]
            file_name = "{}_{}_index_{}".format(self.dataset_name, (index + 1), index)
            lf.basicplot(fig_vector, save_im=save_im, disp_im=disp_im, file_name=file_name)

    def objectarea(self, show_out=True, save_out=False):
        """Determine total surface area per figure per area-numerosity combination for given data set.

        Parameters:
        :param show_out: whether to show the output in the terminal (default: True)
        :param save_out: whether to save the output in a file (default: False)

        """
        # for each area:
        #   for each numerosity:
        #       for each figure:
        #       calculate relative surface area of objects
        #   calculate mean of all figures per area-numerosity combination
        #   calculate the intended relative surface area
        #   print results
        # return array with information

        av_surf_list = []                  # List to store average surface areas

        for area in self.area_list:
            for numerosity in self.numerosities_list:

                area_index = area / self.dis_area

                starting_point = ((area_index - 1) * self.num_numerosities * self.per_area_numerosity) + \
                                 ((numerosity - 1) * self.per_area_numerosity)
                des_range = [(starting_point + x) for x in range(self.per_area_numerosity)]

                area_surf_list = self.surf_area[des_range[0], (des_range[-1] + 1)]
                mean_surf = np.mean(area_surf_list)

                if show_out:
                    sent = 'Area: {}, numerosity: {}, mean area: {}'.format(area, numerosity, mean_surf)
                    print(sent)

                temp_list = [area, numerosity, mean_surf]
                av_surf_list.append(temp_list)

        if save_out:
            pickle.dump(av_surf_list, open('{}_av_surf_list'.format(self.dataset_name)))

        return av_surf_list


if __name__ == '__main__':
    """Call (one of) the functions defined above to plot some figures if script is main script."""
    # Pick any amount between 0 and 51,200:
    how_many = 5
    # Pick an area from [32, 64, 96, 128, 160, 192, 224, 256]:
    this_area = 192
    # Pick a numerosity between 1 and 32:
    this_numerosity = 1
    
    PlotTraining1 = FigurePlotter('training1')
    
    PlotTraining1.plotrandomfigures(how_many)
    
    # figure_object.plotareafigures(how_many, this_area)
    
    # figure_object.plotnumerosityfigures(how_many, this_numerosity)
    
    # figure_object.plotcertainfigures(how_many, this_area, this_numerosity)
    
    '''If applicable: load vector containing desired indices'''
    # indices = pickle.load( open( 'indices', 'rb') )
    
    # figure_object.plotthesefigures(indices)
    
    # area_info_array = figure_object.objectarea()
