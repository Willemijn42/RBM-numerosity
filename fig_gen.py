"""
Produce training or testing data for the RBM. Can be used on its own, or imported in another program as a module.
Depending on that it will produce pickled output files or return values in a dictionary:

- the most important one is called 'data_array', dimensions 51,200 x 900
- the second is 'data_array_info', a list of lists with for each figure the image number, area and numerosity, resp.
- the third is 'obj_info', a list of arrays containing the dimensions and coordinates of all objects of all figures
- the fourth is 'parameter_list', containing some information about the initial parameter settings of the run
- the last is 'surf_area', a list containing the total surface area of each figure

Each line consists of a vector of 900 zeros and ones, representing a training image for the RBM model.
The individual training images are created by adding one object at a time.
Dimensions and coordinates for the objects are randomly generated.
Objects must not overlap, and have to be separated by a buffer. These conditions are checked prior to insertion.

NOTE: this program assumes that the distance between areas is equal to the smallest area (most likely equal to 32).
"""


import pickle
import math as mt
import numpy as np
import numpy.random as elrandom


def generatefigures(data_stage='training1', num_areas=8, area_dis=32, num_numerosities=32, num_fig=200,
                    pix_buffer=1, noise_lev1=0.15, noise_lev2=0.3, dim_fig=30, pickle_data=True):
    """Generate training and testing data for the RBM.

    Parameters:
    :param data_stage: function of the data to be produced (default: 'training1')
    :param num_areas: number of area levels (default: 8)
    :param area_dis: distance between areas in pixels (default: 32)
    :param num_numerosities: range of numerosities (default: 32)
    :param num_fig: number of figures to be generated per area - numerosity combination (default: 200)
    :param pix_buffer: buffer between objects in the figures (default: 1)
    :param noise_lev1: first noise level in generating object dimensions (default: 0.15)
    :param noise_lev2: second noise level in generating object dimensions (default: 0.3)
    :param dim_fig: dimensions in pixels of the (square) figures to be generated (default: 30)
    :param pickle_data: whether to pickle data or return it (default: True)
    :return: file or dictionary with data figures and supplementary information, depending on variable pickle_data.
     Dictionary elements:
     'data_array': array containing the figures as rows, shape: (num_fig_tot, dim_fig^2)
     'data_array_info': list of lists containing information about the dataset per figure [fig_index, area, numerosity]
     'obj_info': list of arrays. One array per figure, shape: [fig_numerosity, 4]. Each row of the array contains the
      dimensions and coordinates of an object: (obj_width, obj_height, obj_x, obj_y).
     'parameter_list': list containing information about the dataset at large: [num_areas (int), num_numerosities (int),
      num_fig (int), area_dis (int), areas (list), numerosities (list)
     'surf_area': list with length num_fig_tot containing the total surface area of all objects per figure in pixels
    """

    areas = [((x + 1) * area_dis) for x in range(num_areas)]            # List of cumulative surface areas [32-256]
    # (actual surface areas will vary slightly because of gaussian noise added in loop below)
    numerosities = [(x + 1) for x in range(num_numerosities)]           # Create list of numerosities [1-32]
    
    num_fig_tot = num_areas * num_numerosities * num_fig        # Total number of images to be created
    fig_counter = 0                                             # Figure counter, up to (num_fig_tot-1) (=51,200-1)
    dim_fig_tot = dim_fig**2                                    # Get length of input vector
    # Some parameters for future use
    parameter_list = [num_areas, num_numerosities, num_fig, area_dis, areas, numerosities]
    
    while_limit = 200                               # Set maximum times new coordinates are generated before quitting
    while_limit2 = 300                              # Set maximum times all object coordinates are reset before quitting
    
    data_array = np.zeros((num_fig_tot, dim_fig_tot))         # Make data array to insert images in, then to be pickled
    data_array_info = []                            # Empty list to append figure information to. Becomes list of lists
    obj_info = []                                   # Empty list to append all object info to, becomes list of arrays
    surf_area = []                                  # Empty list to append total surface area per figure to

    """Generate object dimensions and coordinates"""
    for area in areas:                                                  # Cycle over cumulative surface area levels
        
        for numerosity in numerosities:                                 # Cycle over numerosity levels
        
            for _ in range(num_fig):                                    # Cycle over number of figures per combination
                
                counter2 = 0                                            # Infinite while loop protections

                """Loop to assign dimensions to each object"""
                while counter2 < while_limit2:
                    
                    counter3 = 0                                        # Infinite while loop protection
                    
                    while counter3 < while_limit:
                        
                        res_area = area                                 # Get variable residual area
                        
                        fig_array = np.zeros((dim_fig, dim_fig))    # Make figure array to add obj_ones to
                        obj_array = np.zeros((numerosity, 4))       # Make object array to add dim and coordinates to:
                        # [x, :2] = dimensions, [x, 2:] = left upper corner. Mostly for debugging.

                        res_obj = numerosity
                        
                        for item in range(numerosity):
                            
                            res_obj = numerosity - item
                            obj_noise = elrandom.normal(0, noise_lev1, 1)       # Create first layer of noise
                            obj_area = (float(res_area) / res_obj) + obj_noise  # Calculate area plus first noise layer
                            
                            if obj_area <= 0:  # Exit for loop and restart while loop if obj area is zero to avoid error
                                counter3 += 1
                                break
                            
                            obj_dim = mt.sqrt(obj_area)       # Take root of area to obtain initial object dimensions
                            list_dim = [obj_dim] * 2          # Create list with two initial positive dimensions

                            """While loop to avoid negative or zero dimensions"""
                            while True:
                                dim_noise = elrandom.normal(0, noise_lev2, 2)
                                # Create random noise array to add to object dimensions
                                dim = list_dim + dim_noise           # Array 'dim' contains actual dimensions of object
                                dim_round = np.around(dim, 0)        # Round the dimensions
                                
                                """If all dimensions are greater than zero the while loop can be exited
                                NOTE: CAN CAUSE A SLIGHT BIAS TOWARDS LARGER CUMULATIVE SURFACE AREAS"""
                                if all(x > 0 for x in dim_round):
                                    break
                            
                            obj_array[item, :2] = dim_round           # Insert rounded dimensions into the object array
                            
                            act_area = dim_round[0] * dim_round[1]    # Calculate actual surface area
                            res_area -= act_area                      # Subtract actual object area from total area
                            
                            if res_obj == 1:               # Way to communicate all dimensions are assigned successfully
                                res_obj = 0

                        """If all objects are assigned successfully in for loop above, dimensions while loop can exit.
                        Otherwise initiate next round of dimension assignment."""
                        if res_obj == 0:
                            break
                        
                        if counter3 == while_limit:               # Let's check if this ever happens with these settings
                            print('Failed dimension assignment.')

                    obj_placed = 0                           # To keep track of all objects that are successfully placed
                    
                    """Now that all objects have dimensions, generate object coordinates one by one"""
                    for item in range(numerosity):
                        
                        counter1 = 0                              # Used to avoid infinite while Loop
                        
                        """Set some parameter used in this section"""
                        obj_width = obj_array[item, 0]
                        obj_height = obj_array[item, 1]
                        x_limit = dim_fig - obj_width
                        y_limit = dim_fig - obj_height
                        
                        # obj_area = obj_width * obj_height           # Used in debugging
                        
                        while counter1 < while_limit:
                            
                            # Generate random x-coordinate for left upper corner:
                            obj_array[item, 2] = elrandom.randint(0, (x_limit + 1))
                            obj_x = obj_array[item, 2]
                            # Generate random y-coordinate for left upper corner:
                            obj_array[item, 3] = elrandom.randint(0, (y_limit + 1))
                            obj_y = obj_array[item, 3]
                            # Object array is not really used for anything else after this, handy for debugging?
                            
                            # Make ones array in shape of object:
                            obj_ones = np.ones((obj_width, obj_height))

                            """Check if the assigned space in the figure array is available"""
                            
                            # Determine size of pixel buffer:
                            # X:
                            if obj_x == 0 or obj_x == (dim_fig - obj_width):
                                x_buffer = 1 * pix_buffer
                            else:
                                x_buffer = 2 * pix_buffer
                            # Y:
                            if obj_y == 0 or obj_y == (dim_fig - obj_height):
                                y_buffer = 1 * pix_buffer
                            else:
                                y_buffer = 2 * pix_buffer
                            
                            # Make zeros array shape of obj plus buffer, used to check availability of assigned space:
                            obj_zeros_width = int(obj_width + x_buffer)
                            obj_zeros_height = int(obj_height + y_buffer)
                            obj_zeros = np.zeros((obj_zeros_width, obj_zeros_height))
                            
                            # Assign coordinates to zeros array (= size of area to be checked).
                            # Coordinates depend on location of object (whether it is right at the edges of the figure)
                            if obj_x == 0:                          # X
                                x_obj_zeros = obj_x
                            else:
                                x_obj_zeros = obj_x - 1
                            
                            if obj_y == 0:                          # Y
                                y_obj_zeros = obj_y
                            else:
                                y_obj_zeros = obj_y - 1
                            
                            # Check if the space with buffer is available:
                            # For better overview:
                            obj_z_left = int(x_obj_zeros)
                            obj_z_right = int(x_obj_zeros + obj_zeros_width)
                            obj_z_up = int(y_obj_zeros)
                            obj_z_down = int(y_obj_zeros + obj_zeros_height)
                            
                            # Variable representing the target area in the figure array:
                            test_area = fig_array[obj_z_left:obj_z_right, obj_z_up:obj_z_down]

                            """If space is available, insert ones and move on to next object.
                            If not, loop again and get new coordinates to try."""
                            if (obj_zeros == test_area).all():
                                # If so, generated coordinates can be used:
                                obj_array[item, 2:4] = [obj_x, obj_y]   # Fill obj array with coordinates for debugging
                                
                                # For better overview:
                                obj_left = int(obj_x)
                                obj_right = int(obj_x + obj_width)
                                obj_up = int(obj_y)
                                obj_down = int(obj_y + obj_height)
                                
                                # Insert ones in the location of the object the figure array:
                                fig_array[obj_left:obj_right, obj_up:obj_down] = obj_ones
                                
                                # Once valid location is found and object is inserted, record that and exit while loop:
                                obj_placed += 1
                                break

                            else:
                                """If the space is not available, add 1 to while counter and do while loop again,
                                so new coordinates are generated and checked for this particular object."""
                                counter1 += 1                       # Loop will exit automatically once limit is reached

                    """If all objects are placed we can exit the while loop and store the object information"""
                    if obj_placed == numerosity:
                        new_info = [fig_counter, area, numerosity]
                        data_array_info.append(new_info)
                        obj_info.append(obj_array)
                        # print('Figure: %d, Area: %d, Numerosity: %d, Objects placed: %d'
                        # % ((fig_counter+1), area, numerosity, obj_placed))
                        break

                    """If not, the while loop needs to be executed again to generate new coordinates for all objects
                    in this particular image"""
                    counter2 += 1                                       # Infinite while loop protection 2
                    if counter2 == while_limit2:                        # Used in debugging
                        print('Failed figure.')
                        print('Area = {}, numerosity = {}'.format(area, numerosity))
                        print('Figure index = {}, objects placed = {}'.format(fig_counter, obj_placed))

                """Add complete figures to final array, and pickle final array"""
                fig_list = fig_array.flatten()                          # Flatten array to obtain vectorized figure
                data_array[fig_counter, :] = fig_list                    # Add obtained vector as row to final array
                surf_area.append(sum(fig_list))                         # Store surface area of this figure
                fig_counter += 1
                
    # Create main dictionary with all information
    return_dict = {'data_array': data_array, 'data_array_info': data_array_info, 'obj_info': obj_info,
                   'parameter_list': parameter_list, 'surf_area': surf_area}
    
    """Depending on pickle_data: pickle output file, otherwise return."""
    if pickle_data:
        pickle.dump(return_dict, open(data_stage, 'wb'))
    else:
        return return_dict
