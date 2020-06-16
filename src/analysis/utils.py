'''
Created on 16.08.2018

@author: larissa.hoefling@uni-tuebingen.de
'''
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from copy import deepcopy
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from analysis.spikes import Spikes
from analysis.lightResponse import LightResponse
import matplotlib.ticker as ticker
import matplotlib as mpl


fontpropsLarge = fm.FontProperties(size = 'large', weight='bold')

def znorm(self, y_data):
    """
    Z-transform data
    """

    transformed_array = (y_data - y_data.mean()) / y_data.std()
    return transformed_array


def sigmoid(x, x0, y_max, y_min, k):
    try:
        y = y_max / (1+np.exp(-k*(x-x0)))+ y_min
    except(RuntimeWarning):
        print(x)
    return y


def exponential(x, a, b, c):
    y = a*np.exp(b*x)+c
    return y


def test_goodness_of_fit( y, y_est):
    
        
    total_var = np.sum((y - np.mean(y))**2)
    res_var = np.sum((y-y_est)**2)
    r_squared = 1 - res_var/total_var
    return r_squared


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError
#     , "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError
#     "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError
#     , "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]] # pad the signal
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]


def find_matching_cells(cell_loc_light, cell_loc_el, max_dist=1.5, verbose = False):
    '''
    This function creates a distance matrix between cells found in the light
    recording and cells found in the electrical stimulation recording
    
    Inputs:
    ---------------------------------------------------------------------------
    cell_loc_light    dictionary
                      cell IDs as keys, a tuple of (col, row) as values
    cell_loc_el        dictionary
                       cell ID as keys, a tuple of (col, row) as values
    my_dist            int
                        maximum allowed distance (in electrodes) for two cells 
                        detected in electrical and light recording to be 
                        considered the same
                    
    Outputs:
    ---------------------------------------------------------------------------
    distance_df    pandas dataframe
                   row indices are cell IDs from light recording, column indices
                   are cell IDs from electrical recording, values are distances
                   between the cells (actual distance d if d<max_dist, otherwise
                   infinity)
    matched_cells    dictionary
                     electrical cell IDs as keys, matched light cell IDs as values
                     if a match is found, otherwise None
    '''


    cell_light_list = list(cell_loc_light.keys())
    cell_el_list = list(cell_loc_el.keys())
    no_light_cells = len(cell_light_list)
    no_el_cells = len(cell_el_list)
    distance_matrix = np.full((no_light_cells, no_el_cells), np.nan)
    distance_df = pd.DataFrame(distance_matrix, index = cell_loc_light.keys(), columns = cell_loc_el.keys())
    
    
    for idx_el, el_loc in cell_loc_el.items():
        for idx_light, light_loc in cell_loc_light.items():
            distance_df.loc[idx_light,idx_el]  = tuple_distance(el_loc, light_loc)[0]
    for el_cell in cell_el_list:
        # get the cell in the light recording that is closest to el_cell
        min_dist_idx = distance_df.loc[:,el_cell].idxmin()
        min_dist = distance_df.loc[min_dist_idx,el_cell]
        check = True
        counter = 0
        while (check and counter<no_light_cells):
            #check whether there is a cell in the electrical recording which is
            #closer to the minimum distance light recording cell
            if min_dist < \
            distance_df.loc[min_dist_idx,np.asarray(cell_el_list)!=el_cell].values.min():
                # if that is not the case, set all entries in the el_cell column
                #except for the min_dist_idx to infinity
                
                distance_df.loc[:, el_cell] = \
                np.full_like(distance_df.loc[:,el_cell], np.inf)
                distance_df.loc[min_dist_idx, el_cell] = min_dist
                check = False
                counter+=1
            else:
                #if it is the case, then set the entry in the distance matrix
                # to infinity and check the second closest light cell 
                distance_df.loc[min_dist_idx, el_cell] = np.inf
                min_dist_idx = distance_df.loc[:, el_cell].idxmin()
                min_dist = distance_df.loc[min_dist_idx, el_cell]
                counter+=1
    
    distance_df = distance_df.mask(distance_df>max_dist, np.inf)
    
    matched_cells = dict.fromkeys(cell_loc_el.keys())
    for el_cell in cell_loc_el.keys():
        min_dist = distance_df.loc[:, el_cell].min()
        if min_dist<np.inf:
            matched_cells[el_cell] = distance_df.loc[:, el_cell].idxmin()
        else:
            matched_cells[el_cell] = None 
    if verbose:
        for key, value in matched_cells.items():
            print('E-cell name:{}; e-cell location:{}'.format(key, cell_loc_el[key]))
            if value is None:
                print('No match found')
            else:
                print('light-cell name:{}; light-cell location:{}'.format(value, cell_loc_light[value]))
                d = distance_matrix[value-1, key-1]
                print('distance:{}'.format(d))
        
    return distance_df, matched_cells



def tuple_distance(tup1, tup2):
    x_dist = np.abs(tup1[0]-tup2[0])
    y_dist = np.abs(tup1[1] - tup2[1])
    euclid_dist = np.sqrt(x_dist**2 + y_dist**2)
    dist = np.abs(tup1[0]-tup2[0]) + np.abs(tup1[1] - tup2[1])
    
    return euclid_dist, dist

def min_tuple_dist(tups, loc):
    min_dist = np.inf
    min_tup = (np.inf, np.inf)
    for tup in range(len(tups)):
        new_dist,_ = tuple_distance(tups[tup], loc)
        new_tup = tups[tup]
        idx = np.argmin([min_dist, new_dist])
        min_dist = np.min([min_dist, new_dist])
        min_tup = [min_tup, new_tup][idx]
        
    return min_dist, min_tup



def create_light_response_summary(light_response, spikes, opticalFilter=None,\
                                   max_dist = 1.5):
    
    distance_matrix_lr, matched_cells_lr = \
    find_matching_cells(light_response.cell_to_loc, spikes.cell_to_loc, \
                        max_dist = max_dist)


    if opticalFilter is not None:
        distance_matrix_of, matched_cells_of = \
        find_matching_cells(opticalFilter.cell_to_loc, spikes.cell_to_loc) 
        
        
        final_bias = dict.fromkeys(matched_cells_lr.keys())
        optical_filter = dict.fromkeys(matched_cells_lr.keys())
        for key, value in matched_cells_of.items():
            if value is None:
                final_bias[key] = light_response.bias_index.get(matched_cells_lr[key], -2)
                
                optical_filter[key] = None
            else:
                if matched_cells_lr[key] is None:
                    b = opticalFilter.bias_index[opticalFilter.cell_to_loc[value]][0]
                    final_bias[key] = b
                else:
                    distance_of = distance_matrix_of.loc[value, key]
                    distance_lr = distance_matrix_lr.loc[matched_cells_lr[key], key]
                    if distance_lr<distance_of or \
                    opticalFilter.bias_index[opticalFilter.cell_to_loc[value]][0] == 'x':
                    
                        final_bias[key] = light_response.bias_index[matched_cells_lr[key]]
                    else:
                        b = opticalFilter.bias_index[opticalFilter.cell_to_loc[value]][0]
                        final_bias[key] = b 
                    
                optical_filter[key] = opticalFilter.filter[opticalFilter.cell_to_loc[value]]
        
        final_bias = {key: -2 if element =='x' else np.round(float(element),3)\
                       for key, element in final_bias.items() }
        return final_bias, optical_filter, matched_cells_lr, matched_cells_of
    else:
            
        final_bias = dict.fromkeys(matched_cells_lr.keys())
        final_transience = dict.fromkeys(matched_cells_lr.keys())
        for key, value in matched_cells_lr.items():
            final_bias[key] = light_response.bias_index.get(matched_cells_lr[key], -2)
            final_transience[key] = light_response.transience_index.get(matched_cells_lr[key], -5)
        
        final_bias = {key: -2 if element =='x' else np.round(float(element),3)\
                       for key, element in final_bias.items() }
        return final_bias, final_transience, matched_cells_lr
    

def plot_summary(cells, stimulus, el_spikes, sta, matched_cells = None, \
                 bias = None, light_spikes = None, my_glmodel=None,\
                  robust=False, density_light = True, density_el = True,\
                  show_std = False, path=None,fname='default', cmap = 'Blues'):
    
    assert sta.robust == robust, 'Reliability parameter of the STA object is inconsistent with the Reliability parameter that was passed to plot_summary()'
    
    cmap = cmap
    #set general figure settings 
#     fig = plt.figure(figsize = (12,20))
    fig = plt.figure()
    s = 13
    mpl.rcParams.update({'font.size':s, 'axes.labelsize':s,\
                          'xtick.labelsize':s, 'ytick.labelsize':s})
    hist_axes = []
#     plt.style.use('seaborn-paper')
    if light_spikes is not None:
        multiplier = 3
        n_cols = 14
        n_rows = 2+ 1*len(cells)
        col_pos = [0, 2, 8, 10, 12]
        poss = [1*multiplier, 2*multiplier]
        cell_biases = [b for cell, b in bias.items() if cells.count(cell)]
        if hasattr(sta, 'cluster_labels'):
            labels = [sta.cluster_labels[cell] for cell in cells]
            plot_position = np.argsort(list(labels))
        else:
            plot_position = np.argsort(cell_biases)
    else:
        multiplier = 6
        n_cols = 14
        col_pos=[0, 6, 8, 10]
        poss = [0, 1*multiplier]
        n_rows = len(cells) + 2
        
        if hasattr(sta, 'cluster_labels'):
            labels = [l for cell, l in sta.cluster_labels.items() if cells.count(cell)]

            plot_position = np.argsort(list(labels))
        else:
            plot_position = np.linspace(0,len(cells)-1, len(cells))
    if light_spikes is not None:
        '''
        ###########################################################################
        plot the light stimulus (step)
        '''
        
        ax = plt.subplot2grid([n_rows, n_cols], [0, col_pos[0]], colspan=2, rowspan=2)
        x_light_stim = np.linspace(0, 1, 1000)
        y_light_stim = np.concatenate([np.zeros(500), np.ones(499), [0]])
        plt.plot(x_light_stim, y_light_stim, color = 'k', label = 'light stimulus')
#         plt.annotate('Light off', (0.1, 0.2))
#         plt.annotate('Light on', (0.6, 0.2))
        ax.set_xlim([0,1])
#         ax.set_xlabel('Time [s]')
#         ax.xaxis.set_label_position('top')
    '''
    ###########################################################################
    plot first 1000 ms of electrical stimulus
    '''    
    if light_spikes is not None:
        
        stop = 30000
        ax_ref_el = plt.subplot2grid([n_rows, n_cols], [0, col_pos[1]], rowspan=2, colspan=6)
        x_el_stim = stimulus.time_line[:stop]*10**-6 #convert from us to s
        y_el_stim = stimulus.current_density[:stop]
        plt.plot(x_el_stim, y_el_stim, color = 'k', label = 'el. stimulus', \
                 linewidth = 0.5)
        
        transform_obj = ax_ref_el.transData
        
        horizontal_scalebar = AnchoredSizeBar(transform_obj, 0.1,\
                           label = '', loc='upper center', 
                           borderpad=-1, label_top=True,
                           frameon=False,
                           size_vertical=0.1
                           )
#         ax_ref_el.add_artist(horizontal_scalebar)
        
        vertical_scalebar = AnchoredSizeBar(transform_obj, 0.005,\
#                            label=r'1 $\frac{mA}{cm^2}$', 
                            label = '',
                           loc='center right', 
                           borderpad=-1, label_top=True,
                           frameon=False, size_vertical=1
                           )
#         ax_ref_el.add_artist(vertical_scalebar)    
    
    else:
        stop = 30000
        ax_ref_el = plt.subplot2grid([n_rows, n_cols], [0, col_pos[0]], rowspan=2, colspan=6)
        x_el_stim = stimulus.time_line[:stop]*10**-6 #convert from us to s
        y_el_stim = stimulus.current_density[:stop]
        plt.plot(x_el_stim, y_el_stim, color = 'k', label = 'el. stimulus', \
                 linewidth = 0.5)
        
        transform_obj = ax_ref_el.transData
        
        horizontal_scalebar = AnchoredSizeBar(transform_obj, 0.1,\
                           '0.1 s', 'upper center', 
                           borderpad=-1, label_top=True,
                           frameon=False
                           )
#         ax_ref_el.add_artist(horizontal_scalebar)
        
        vertical_scalebar = AnchoredSizeBar(transform_obj, 0.001,\
                           r'1 $\frac{mA}{cm^2}$', 'upper right', 
                           borderpad=-1, label_top=True,
                           frameon=False, size_vertical=1
                           )
#         ax_ref_el.add_artist(vertical_scalebar)    
    
    

    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
    # exception because of the number of lines being plotted on it.    
    
    '''
    ###########################################################################
    initialize a subplot that serves the purpose to display xticks

    
    ''' 
    ax_ref_sta = plt.subplot2grid([n_rows, n_cols], [0,col_pos[2]], colspan=2)
    
    plt.gca().set_xlim(-0.02, 0.01)
    for label in ax_ref_sta.get_yticklabels():
        label.set_visible(False)
    ax_ref_sta.tick_params(axis = 'both', which ='both', bottom = False, left = False)
    
    plt.subplot2grid([n_rows, n_cols], [0,10], colspan=2)
    
    plt.gca().set_xlim(-0.02, 0.01)
    for label in ax_ref_sta.get_yticklabels():
        label.set_visible(False)
    plt.gca().tick_params(axis = 'both', which ='both', bottom = False, left = False)

    
    for i, cell in enumerate(cells):
        
        pp = np.where(plot_position == i)[0][0]+1
        print(cell, pp)
        if light_spikes is not None:
            
            
        
            '''
            ###########################################################################
            plot the light response raster rasters for all cells
            
            ''' 
            
            ax = plt.subplot2grid([n_rows, n_cols], [pp+1,col_pos[0]], colspan=2)
            
            if matched_cells[cell] is not None:
                light_spikes.raster(matched_cells[cell], ax = ax, \
                                    density = density_light,cmap = cmap)
                (ymin, ymax) = ax.get_ylim()
                plt.vlines(0.5, ymin, ymax, color = 'r', linewidths = 0.5 )

            leftmost_ax = ax
            
            

        
            '''
            ########################plot electrical raster#########################
            '''
            ax = plt.subplot2grid([n_rows, n_cols], [pp+1, col_pos[1]], colspan=6, \
                                  sharex = ax_ref_el)
            el_spikes.raster(cell, ax = ax, ms = True, stop = int(stop/10000), robust=robust, \
                            density = density_el, raster_show = True, cmap = cmap)

            
        else:
            ax = plt.subplot2grid([n_rows, n_cols], [pp+1, col_pos[0]], colspan=6, \
                                  sharex = ax_ref_el)
            el_spikes.raster(cell, ax = ax, ms = True, stop = int(stop/10000), robust=robust, \
                             density = density_el, cmap = cmap)
            
            leftmost_ax = ax
        

        leftmost_ax.yaxis.labelpad = 20
        leftmost_ax.set_ylabel(str(np.round(np.average(el_spikes.benchmark[cell]), 2)), \
                               rotation = 'horizontal', verticalalignment = 'center',
                               fontsize= 8)
        
        
        '''
        ########################plot STA#########################
        '''
        if light_spikes is not None:
            ax = plt.subplot2grid([n_rows, n_cols], [pp+1, col_pos[2]], colspan=2,\
                               sharex = ax_ref_sta)
        else:
            ax = plt.subplot2grid([n_rows, n_cols], [pp+1, col_pos[1]], colspan=2,\
                               sharex = ax_ref_sta)
        
        x = np.linspace(-0.02,0.01,sta.len_sta_samples)    
              
#         sta_white = np.average(sta.sta[cell], 0)
#         std = np.average(sta.ste_stand_dev[cell], 0)
        sta_white = np.average(sta.full_sta[cell], 0)
        std = np.std(np.concatenate(sta.full_ste[cell], axis=1), axis=1)
        cl = sta.color_mapping[sta.colors_dict[cell]][0]
        plt.plot(x, sta_white, color = cl)
        if show_std:
            plt.fill_between(x, sta_white-std, sta_white+std, color = cl, alpha = 0.4)
#         plt.plot(x, np.zeros_like(x), linestyle = '--', color = 'k')

        (ymin, ymax) = ax.get_ylim()
        plt.vlines(0, -0.2, 0.2, color = 'k', linewidth = 0.7)
        plt.hlines(0, xmin=-0.02, xmax=0.01, linestyles='--', linewidths = 0.5)


        if pp == len(plot_position)-1:
            transform_obj = ax.transData
            horizontal_scalebar = AnchoredSizeBar(transform_obj, 0.005,\
                                label='5 ms', 
#                                 label='',
                               loc='lower center', 
                               borderpad=-0.5, label_top=False,
                               frameon=False, 
                               size_vertical=0.01
                               )
            
            vertical_scalebar = AnchoredSizeBar(transform_obj, 0.0001,\
                               r'0.4 $\frac{mA}{cm^2}$', 'upper right', 
                               frameon=False, borderpad=-1, label_top=True,
                               size_vertical=0.4
                               )
#             ax.add_artist(horizontal_scalebar)
#             ax.add_artist(vertical_scalebar)
#         
        
        
        if my_glmodel is not None:
            
            if light_spikes is not None:
                ax = plt.subplot2grid([n_rows, n_cols], [pp+1,col_pos[3]], colspan=2,\
                               sharex = ax_ref_sta)
                cl = my_glmodel.color_mapping[my_glmodel.colors_dict[cell]][0]
                
            else:
                ax = plt.subplot2grid([n_rows, n_cols], [pp+1,col_pos[2]], colspan=2,\
                               sharex = ax_ref_sta)
                if hasattr(my_glmodel, 'cluster_labels'):
                    cl = my_glmodel.color_mapping[my_glmodel.colors_dict[cell]][0]
                else:
                    cl = 'gray'
            all_betas = [my_glmodel.models[cell][i].coef_ for i in
                         range(my_glmodel.n_cross_validation)]
            beta = np.squeeze(np.average(all_betas, axis=0))
            plt.plot(x, beta, color = cl)
            (ymin, ymax) = ax.get_ylim()
            plt.vlines(0, -0.05, 0.05, color = 'k', linewidth = 0.7)
            plt.hlines(0, xmin=-0.02, xmax=0.01, linestyles='--', linewidths = 0.5)
        
    if light_spikes is not None:
        axes_break = 4
    else:
        axes_break = 3
        
    for i, ax in enumerate(fig.get_axes()[:-axes_break]):
        # Remove the plot frame lines. They are unnecessary chartjunk. 
        
        ax.xaxis.tick_top()   
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False)    
        ax.tick_params(axis = 'both', which ='both', bottom = False, \
                           left = False, top = False, right = False)
        
        for label in ax.get_xticklabels():
            label.set_visible(False)
            
        for label in ax.get_yticklabels():
            label.set_visible(False)
            

    
    fig.get_axes()[-4].xaxis.set_major_locator(plt.MaxNLocator(1))
    fig.get_axes()[-3].xaxis.set_major_locator(plt.MaxNLocator(4))
    fig.get_axes()[-2].xaxis.set_major_locator(plt.MaxNLocator(2))
    fig.get_axes()[-1].xaxis.set_major_locator(plt.MaxNLocator(2))
#     fig.get_axes()[-1].xaxis.set_major_locator(plt.MaxNLocator(2))
#     fig.get_axes()[-3].set_xlabel('Time [s]', horizontalalignment = 'right')
    fig.text(0.35, 0.02, 'Time [s]', va = 'center', fontsize = 10)
    for ax in fig.get_axes()[-axes_break:]:
        # Remove the plot frame lines. They are unnecessary chartjunk. 
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False) 
#         ax.xaxis.set_major_locator(plt.MaxNLocator(3))   
#         ax.tick_params(axis = 'both', which ='both', bottom = False, \
#                            left = False, top = False, right = False)
    
        ax.tick_params(axis = 'both', which ='both',  \
                           left = False, top = False, right = False)
        for label in ax.get_xticklabels():
            label.set_fontsize(8)
            
        for label in ax.get_yticklabels():
            label.set_visible(False)
                
    
        
    plt.subplots_adjust(hspace = 0.2, wspace=0.5)
    if path is not None:
#         plt.savefig(path+fname+'.ps', dpi = 300, fmt='ps')
        plt.savefig(path+fname+'.png', dpi = 300, fmt='png')
        plt.savefig(path+fname+'.pdf', dpi = 300, fmt='pdf')
        plt.savefig(path+fname+'.svg', dpi = 300, fmt='svg')
    else:
        
        plt.show()
    plt.close(fig)
        

def plot_summary_el_only(cells, stimulus, el_spikes,  sta,\
                          my_grid_search=None, robust = False):
    
    assert sta.robust == robust, 'robustness parameter of the STA object is inconsistent with the robustness parameter that was passed to plot_summary()'

    #set general figure settings 
    fig = plt.figure()
    n_rows = 1+ 1*len(cells)
    n_cols = 5
    multiplier = 2
    
    max_sn = np.max(np.concatenate(list(sta.sta_norm.values())))
    
    if hasattr(sta, 'cluster_labels'):
        labels = [l for cell, l in sta.cluster_labels.items() if cells.count(cell)]
#         plt.ylabel('id: {} , cl: {}'.format(cell, sta.cluster_labels[cell]) ,\
#                     rotation = 'horizontal')
        plot_position = np.argsort(list(labels))
    else:
        plot_position = np.linspace(0,len(cells)-1, len(cells))
    
    '''
    ###########################################################################
    plot first 1000 ms of electrical stimulus
    '''    
    plt.subplot2grid([n_rows, n_cols], [0,0*multiplier], colspan=2)
    x_el_stim = stimulus.time_line[:10000]*10**-6 #convert from us to s
    y_el_stim = stimulus.current_density[:10000]
    plt.plot(x_el_stim, y_el_stim, color = 'k', label = 'el. stimulus')
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
    # exception because of the number of lines being plotted on it.    
    
    '''
    ###########################################################################
    initialize a subplot that serves the purpose to display xticks

    
    ''' 
    ax_ref = plt.subplot2grid([n_rows, n_cols], [0,1*multiplier], colspan=2)
    
    plt.gca().set_xlim(-0.02, 0.01)
    
    for i, cell in enumerate(cells):
        '''
        ########################plot electrical raster#########################
        '''
        pp = np.where(plot_position == i)[0][0]
        ax = plt.subplot2grid([n_rows, n_cols], [pp+1, 0*multiplier], colspan=2)
        el_spikes.raster(cell, ax = ax, ms = True, stop = 1, robust=robust)
        for label in ax.get_xticklabels():
            label.set_visible(False)
        for label in ax.get_yticklabels():
            label.set_visible(False)
        
        ax.tick_params(axis = 'both', which ='both', bottom = False, left = False)
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False) 
        if hasattr(sta, 'cluster_labels'):
#             plt.ylabel(str(sta.cluster_labels[cell]), rotation = 'horizontal')
         
            plt.ylabel('id: {} , cl: {}'.format(cell, sta.cluster_labels[cell]) ,\
                    rotation = 'horizontal')
       
        ax.yaxis.labelpad = 70
        '''
        ########################plot STA#########################
        '''
        ax = plt.subplot2grid([n_rows, n_cols], [pp+1, 1*multiplier], colspan=2, sharex = ax_ref)
        
        x = np.linspace(-0.02,0.01,sta.len_sta_samples)    
#         ax.set_xlim([x[0], x[-1]])              
        sta_white = np.average(sta.sta[cell], 0)
#         std = np.average(sta.ste_stand_dev[cell], 0)
        plt.plot(x, sta_white, color = 'r')
        plt.plot(x, np.zeros_like(x), linestyle = '--', color = 'k')
        (ymin, ymax) = ax.get_ylim()
#         sym_lim = np.max(np.abs([ymin, ymax]))
#         ax.set_ylim([-sym_lim, sym_lim])
#         plt.vlines(0, -sym_lim, sym_lim, color = 'k')
        plt.vlines(0, ymin, ymax, color = 'k')
        for label in ax.get_xticklabels():
            label.set_visible(False)
#         for label in ax.get_yticklabels():
#             label.set_visible(False)
        ax.get_yaxis().tick_right()
        ax.tick_params(axis = 'both', which ='both', bottom = False, left = False)
        ax.spines["top"].set_visible(False)    
#         ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False) 
#         plt.fill_between(x, sta_white-std, sta_white+std, color = '0.5', alpha = 0.2)   
        if my_grid_search is not None:
            lambda_idx = int(my_grid_search.best_lambda_idx[cell])
            filter = my_grid_search.best_models[cell][lambda_idx].fit_['beta']
            normed_filter = filter/np.max(filter)
            plt.plot(x, normed_filter,\
                      color = 'g')      
#             plt.plot(x, my_grid_search.models[cell][-1][-2].fit_['beta'],\
#                       color = 'g')        
         
         
        '''
        ######################## initialize axis for STA norm histogram #########################
        '''
        ax = plt.subplot2grid([n_rows, n_cols], [0, 4])
        ax.set_xlim(0, max_sn)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)    
        '''
        ######################## plot STA norm histogram #########################
        '''
        ax = plt.subplot2grid([n_rows, n_cols], [pp+1, 4])
        plt.hist(sta.sta_norm_dist[cell], color = 'k')
        (ymin, ymax) = ax.get_ylim()
        plt.vlines(sta.sta_norm[cell], ymin, ymax, color = 'r')
        ax.set_xlim(0, max_sn)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for label in ax.get_xticklabels():
            label.set_visible(False)
        for label in ax.get_yticklabels():
            label.set_visible(False)
        ax.get_yaxis().tick_right()
        ax.tick_params(axis = 'both', which ='both', bottom = False, left = False)
    
    for ax in fig.get_axes()[:3]:
        # Remove the plot frame lines. They are unnecessary chartjunk. 
        ax.xaxis.tick_top()   
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False)    
        
        
def plot_population_responses(cells, spikes, sta, bins = 1000, robust =False,\
                               plot_hist_diff = False):
    
    fig = plt.figure()
    if plot_hist_diff:
        n_rows = sta.n_clusters+1
    else:
        n_rows = sta.n_clusters
    n_cols = 4
    multiplier = 1
    '''
    top left
    #######################################################################
    plot e-stim excerpt (1000ms)
    '''
#     ax_ref_psth = plt.subplot2grid([n_rows, n_cols], [0,1*multiplier], colspan=1)
#     x_el_stim = stimulus.time_line[:10000]*10**-6 #convert from us to s
#     y_el_stim = stimulus.current_density[:10000]
#     plt.plot(x_el_stim, y_el_stim, color = 'k', label = 'el. stimulus')
#     ax_ref_psth.xaxis.tick_top()
#     ax_ref_psth.set_xlim(0, 0.5)
    
    '''
    left
    ###########################################################################
    plot the average STA for a cluster
    '''
    axes_sta = []
    axes_sta.append(plt.subplot2grid([n_rows, n_cols], [0, 0]))
    for cluster in range(1, sta.n_clusters):
        new_ax = plt.subplot2grid([n_rows, n_cols], \
                                         [cluster, 0],sharex = axes_sta[0], \
                                         sharey = axes_sta[0])
        axes_sta.append(new_ax)
    
    for ax in axes_sta:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_yaxis().tick_right()
    
    sta_all = {cluster:[] for cluster in range(sta.n_clusters)}
    for i, cell in enumerate(cells):
        cluster_label = sta.cluster_labels[cell]
        if cluster_label>=0:
            ax = axes_sta[cluster_label]
            plt.sca(ax)
            x = np.linspace(-0.02,0.01,sta.len_sta_samples)    
            sta_white = np.average(sta.sta[cell], 0)
            plt.plot(x, sta_white, alpha = 0.2)
            plt.plot(x, np.zeros_like(x), linestyle = '--', color = 'k')
    
    #         for label in ax.get_xticklabels():
    #             label.set_visible(False)
    #         for label in ax.get_yticklabels():
    #             label.set_visible(False)
            ax.get_yaxis().tick_right()
            ax.tick_params(axis = 'both', which ='both', bottom = False, left = False)
            ax.spines["top"].set_visible(False)    
    #         ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(False) 
            
            sta_all[cluster_label].append(sta_white)
            
    for cluster in range(sta.n_clusters):
        
        sta_average = np.average(sta_all[cluster], axis =0)
        plt.sca(axes_sta[cluster])
        plt.plot(x, sta_average, color = 'k')
    
        plt.ylabel(str(cluster))
        
    '''
    middle
    #######################################################################
    plot the true population responses
    '''
    axes_true = []
    axes_shuffled = []
    for cluster in range(sta.n_clusters):
        ax = plt.subplot2grid([n_rows, n_cols], [cluster, 1])
        axes_true.append(ax)
        ax = plt.subplot2grid([n_rows, n_cols], [cluster, 2])
        axes_shuffled.append(ax)
        
    if plot_hist_diff:
        ax = plt.subplot2grid([n_rows, n_cols], [sta.n_clusters, 1])
        axes_true.append(ax)
        
        ax = plt.subplot2grid([n_rows, n_cols], [sta.n_clusters, 2])
        axes_shuffled.append(ax)
        
    true_kld = spikes.population_histogram(cells, sta, axes=axes_true, bins = bins, \
                                robust = robust, x_lim = [0, 0.5])
    spikes.population_histogram(cells, sta, axes=axes_shuffled, bins = bins, 
                                random_shuffle = True, robust = robust, x_lim = [0, 0.5])
    
    kld_dist= spikes.compute_population_KLD(cells, sta, robust = robust)
    colors = ['r', 'g', 'b']
    labels = ['0 vs 1', '0 vs 2', '1 vs 2']
    for n in range(kld_dist.shape[1]): 
        ax = plt.subplot2grid([n_rows, n_cols], [n, 3], colspan=2)
    #     color = ['k', 'k', 'k']
        
        ax.hist(kld_dist[:,n], color = colors[n], histtype = 'step')
        (ymin, ymax) = ax.get_xlim()

        plt.vlines(true_kld[n], ymin, ymax, color = colors[n], label = labels[0])

    
  

    
def find_stim_el(stim_grid, fieldwidth, fieldheight):
    stim_els = []
    for el in range(len(stim_grid)):
        upper_left = stim_grid[el]
        for i in range(0,fieldwidth):
            for j in range(0, fieldheight):
                temp = (upper_left[0]+i, upper_left[1]+j)
                stim_els.append(temp)
                
    return stim_els
                

def join_spikes(spike_paths, light_sorted = None, light_raw = None,\
                 max_dist = 1.5, stim_grids = None):
    '''
    Joins two instances of class Spikes (for joining data from different experiments)
    '''
    if stim_grids is not None:
        stim_els = dict.fromkeys(stim_grids.keys())
    spike_objects = []
    for i, path in enumerate(spike_paths):
        if stim_grids is not None:
            rec = list(stim_grids.keys())[i]
            assert(path.find(rec)>0)
            stim_els[rec] = find_stim_el(stim_grids[rec][0], stim_grids[rec][1],\
                                         stim_grids[rec][2])
        spikes = Spikes(path, isicheck=True, tag_check = True, \
                        stim_els = stim_els[rec])
        spike_objects.append(spikes)
    if (light_sorted is not None) and (light_raw is not None):
        light_spikes = []
        bias = []
        transience = []
        matched_cells = []
        for i, path in enumerate(light_sorted):
            these_light_spikes = \
            LightResponse(path = path, raw_file_path=light_raw[i])
            light_spikes.append(these_light_spikes)
            
            spikes = spike_objects[i]
            b, t, mc = create_light_response_summary(these_light_spikes, spikes, max_dist = max_dist)
            bias.append(b)
            matched_cells.append(mc)
            transience.append(t)
            
        joined_light_spikes = deepcopy(light_spikes[0])
        light_spikes.pop(0)
        
        joined_bias = deepcopy(bias[0])
        bias.pop(0)
        
        joined_transience = deepcopy(transience[0])
        transience.pop(0)
        
        joined_matched_cells = deepcopy(matched_cells[0])
        matched_cells.pop(0)

        for i, ls in enumerate(light_spikes):
            x = (i+1)*100
            [joined_light_spikes.structured_ts.update({cell+x : struct_ts}) for\
             cell, struct_ts in ls.structured_ts.items()]
            [joined_light_spikes.cell_to_loc.update({cell+x : loc})for \
             cell, loc in ls.cell_to_loc.items()]
            [joined_bias.update({cell+x : bi}) for cell, bi in bias[i].items()]
            [joined_transience.update({cell+x : ti}) for cell, ti in transience[i].items()]

            for cell in matched_cells[i]:
                if matched_cells[i][cell] is None:
                    joined_matched_cells.update({cell+x: None})
                else:
                    matched_cell=matched_cells[i][cell] 
                    joined_matched_cells.update({cell+x : matched_cell+x})

    joined_spike_object = deepcopy(spike_objects[0])
    joined_spike_object.joined = True
    joined_spike_object.joined_path = [spike_objects[0].path]
    spike_objects.pop(0)
    for i, spike_object in enumerate(spike_objects):
        x = (i+1)*100
        
        [joined_spike_object.structured_ts.update({cell+x : struct_ts}) for \
         cell, struct_ts in spike_object.structured_ts.items()]
        
        [joined_spike_object.training_set.update({cell+x : train_st}) for \
         cell, train_st in spike_object.training_set.items()]
        
        [joined_spike_object.test_set.update({cell+x : test_st}) for \
         cell, test_st in spike_object.test_set.items()]
        
        [joined_spike_object.robust_spikes.update({cell+x : rob_spikes}) for \
         cell, rob_spikes in spike_object.robust_spikes.items()]
        
        [joined_spike_object.robust_training_set.update({cell+x : rts}) for \
         cell, rts in spike_object.robust_training_set.items()]
        
        [joined_spike_object.robust_test_set.update({cell+x : rts}) for \
         cell, rts in spike_object.robust_test_set.items()]
        
        [joined_spike_object.robustness_ratio.update({cell+x : rr}) for \
         cell, rr in spike_object.robustness_ratio.items()]
        joined_spike_object.joined_path.append(spike_object.path)
        
        [joined_spike_object.cell_to_loc.update({cell+x : loc}) for\
         cell, loc in spike_object.cell_to_loc.items()]
        
        [joined_spike_object.dist_to_stim.update({cell+x : dist})for \
         cell, dist in spike_object.dist_to_stim.items()]
        
        [joined_spike_object.closest_stim_el.update({cell+x : el})for \
         cell, el in spike_object.closest_stim_el.items()]
    
    if light_raw is None:
        return joined_spike_object
    else:
        return joined_spike_object, joined_light_spikes, joined_matched_cells, \
            joined_bias, joined_transience
        
        
def join_bias_information(bias_index, matched_cells, to_add):
    '''
    This script joins the bias index and matched cells dictionaries for two 
    measurements where light response information is available
    
    Inputs:
    ---------------------------------------------------------------------------
    bias_index    list
                  of dictionaries containing cell IDs as keys and bias indices 
                  as values
    matched_cells    list
                     of dictionaries containing cell IDs as keys 
    '''
    joined_bias_index = deepcopy(bias_index[0])
    joined_matched_cells = deepcopy(matched_cells[0])
    bias_index.pop(0)
    matched_cells.pop(0)
    for i, bias_idx in enumerate(bias_index):
        x = to_add[i]
        [joined_bias_index.update({cell+x : value+x}) for cell, value in bias_idx.items()]
        mc = matched_cells[i]
        [joined_matched_cells.update({cell+x : value+x}) for cell, value in mc.items()]
        
    return joined_bias_index


def plot_dist_vs_ratio(spikes,
                       ratio,
                       path=None,
                       bias=None,
                       y_scale='linear',
                       legend=False,
                       ax=None,
                       robustness=0,
                       sta=None
                       ):

    plt.style.use('seaborn-paper')
    s = 13
    mpl.rcParams.update({'font.size': s, 'axes.labelsize': s, \
                         'xtick.labelsize': s, 'ytick.labelsize': s})
    cells = [cell for cell, value in ratio.items() if value is not None]
    dist = [spikes.dist_to_stim[cell] for cell in cells]
    y = [ratio[cell] for cell in cells]
    if robustness>0:
        hue = ['nr' if np.average(spikes.benchmark[cell]) <= robustness else 'r'
               for cell in cells]
        my_pal = {'nr': 'k', 'r': 'r'}
    else:
        hue = ['nr' for cell in cells]
        my_pal = {'nr': 'k', 'r': 'r'}
    if sta is not None:
        hue = [sta.colors_dict[cell] for cell in cells]
        my_pal = {key: value[0] for key, value in sta.color_mapping.items()}
    if bias is not None:
        bias_discretized = ['OFF' if bias[cell] < -0.5
                            else 'ON' if bias[cell] > 0.5
                            else 'ON-OFF' for cell in cells]
        size = ['large' for cell in cells]
        ratio_df = pd.DataFrame(
            {'Distance to edge of stimulation electrode': dist,
             'Log of firing rate ratio': y, 'style': bias_discretized,
             'size': size, 'robustness' : hue}
        )

        if ax is None:
            g = sns.jointplot(dist, y, space=0, marginal_kws=dict(bins=20, color='k'))
            g.ax_joint.cla()
            g.ax_marg_x.cla()
            g.ax_marg_x.set_axis_off()
            sns.scatterplot(x='Distance to edge of stimulation electrode',
                            y='Log of firing rate ratio', style='style',
                            data=ratio_df,
                            ax=g.ax_joint, size='size',
                            sizes={'large': 50}, legend=legend,
                            style_order=['OFF', 'ON-OFF', 'ON'],
                            hue='robustness',
                            palette=my_pal
                            )
            g.ax_joint.axhline(y=0, color='k')
            g.ax_joint.set_yscale(y_scale)
            g.ax_joint.set_ylim(-3.2, 3.5)
        else:
            sns.scatterplot(x='Distance to edge of stimulation electrode',
                            y='Log of firing rate ratio', style='style',
                            data=ratio_df,
                            color='k', sizes={'large': 50}, legend=legend,
                            style_order=['OFF', 'ON-OFF', 'ON'], ax=ax,
                            )
            ax.axhline(y=0, color='k')
            ax.set_yscale(y_scale)
            ax.set_xlabel('')
            ax.set_ylabel('')
    else:

        size = ['large' for cell in cells]
        ratio_df = pd.DataFrame({'Distance to edge of stimulation electrode': dist,
                                'Log of firing rate ratio': y,
                                 'size': size, 'robustness': hue})

        if ax is None:
            g = sns.jointplot(dist, y, space=0, marginal_kws=dict(bins=20, color='k'))
            g.ax_joint.cla()
            g.ax_marg_x.cla()
            g.ax_marg_x.set_axis_off()
            sns.scatterplot(x='Distance to edge of stimulation electrode',
                            y='Log of firing rate ratio',
                            data=ratio_df,
                            ax=g.ax_joint, size='size',
                            sizes={'large' : 50}, legend=legend,
                            hue='robustness',
                            palette=my_pal,
                            )
            g.ax_joint.axhline(y=0, color='k')
            g.ax_joint.set_yscale(y_scale)
            g.ax_joint.set_ylim(-3.2, 3.5)
        else:
            sns.scatterplot(x='Distance to edge of stimulation electrode',
                            y='Log of firing rate ratio',
                            data=ratio_df,
                            size='size',
                            color='k', sizes={'large': 50}, legend=legend, ax=ax,
                            )
            ax.axhline(y=0, color='k')
            ax.set_yscale(y_scale)
            ax.set_xlabel('')
            ax.set_ylabel('')

    if path is not None:
        fname = 'dist_vs_ratio_'+y_scale
        plt.savefig(path + fname + '.png', dpi=300, fmt='png')
        plt.savefig(path + fname + '.pdf', dpi=300, fmt='pdf')
        plt.savefig(path + fname + '.svg', dpi=300, fmt='svg')

def plot_dist_vs_benchmark(spikes,
                           sta,
                           cells,
                           path=None,
                           robust_thr=0.15,
                           bias=None,
                           markers=True,
                           barplots=False,
                           bias_thresh=[-0.25, 0.5]):

    f = plt.figure()
    plt.style.use('seaborn-paper')
    s = 13
    mpl.rcParams.update({'font.size':s, 'axes.labelsize':s,\
                          'xtick.labelsize':s, 'ytick.labelsize':s})
    
    b = [np.average(spikes.benchmark[cell], axis = 0) for cell in cells]
    dist = [spikes.dist_to_stim[cell] for cell in cells]
    g = sns.jointplot(dist, b,space = 0, marginal_kws = dict(bins = 20, \
                                                              color = 'k'))
    g.ax_joint.cla()
    g.ax_marg_x.cla()
    g.ax_marg_x.set_axis_off()
    x = [spikes.dist_to_stim[cell] for cell in cells]
    y = [np.average(spikes.benchmark[cell], axis=0) for cell in cells]
    if bias is not None:
        bias_discretized = [-1 if bias[cell] < -1
                            else 0 if bias[cell] < bias_thresh[0]
                            else 2 if bias[cell] > bias_thresh[1]
                            else 1 for cell in cells]
    else:
        if markers:
            bias_discretized = [1 if cell//100 == 2 else 0 for cell in cells]
            print(bias_discretized)
        else:
            bias_discretized = [0 for cell in cells]
    dark_green = sns.color_palette('PiYG', n_colors=10)[-1]
    dark_red = sns.color_palette('YlOrRd', n_colors=10)[-1]
    dark_purple = sns.color_palette('PuOr', n_colors=10)[-1]

    if False:
        hue = [sta.cluster_labels.get(cell, 3) for cell in cells]
    else:
        hue = [4 if np.average(spikes.benchmark[cell], axis=0)>robust_thr\
               else 3 for cell in cells]
    my_pal = {0: dark_green, 1: dark_red, 2:dark_purple, 3:'gray', 4:'black'}
#     my_markers = {0: '.', 1: 's', 2: 'd'}
    my_markers = {'X', 'd', 'o', 's'}
    d = {'Distance to edge of stimulation electrode' : x, \
         'Robustness Index' : y, 'hue' : hue, 'style' : bias_discretized}
    df = pd.DataFrame(d)
    
    sns.scatterplot(x = 'Distance to edge of stimulation electrode', \
                           y = 'Robustness Index',hue='hue', style = 'style',\
                           data = df, \
                           palette = my_pal, ax = g.ax_joint,
                            #legend=False,\
                           markers = my_markers
                    )
    g.ax_joint.axhline(robust_thr, 0, 1, color = 'k')
    g.ax_joint.set_xlabel(r'Distance to edge of stimulation electrode [$\mu m$]', fontsize=s)
    g.ax_joint.set_ylabel('Reliability Index', fontsize=s)
    g.ax_joint.tick_params(axis="both", labelsize=s)
    g.ax_joint.yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.tight_layout()
    fname = 'dist_vs_benchmark'
    if path is not None:
        plt.savefig(path+fname+'.png', dpi = 300, fmt='png')
        plt.savefig(path+fname+'.pdf', dpi = 300, fmt='pdf')
        plt.savefig(path+fname+'.svg', dpi = 300, fmt='svg')
#     plt.show()
    #plt.close('all')

    if barplots:
        f = plt.figure()
        f.add_subplot(2,2,3)
        hue = [sta.cluster_labels.get(cell, 3) for cell in cells]
        d = {'Distance to edge of stimulation electrode' : x, \
             'Robustness Index' : y, 'hue' : hue, 'style' : bias_discretized}
        df = pd.DataFrame(d)


        x = [df.loc[lambda df: df.hue ==0, :]\
                ['Distance to edge of stimulation electrode'],
            df.loc[lambda df: df.hue ==1, :]\
                ['Distance to edge of stimulation electrode'],
            df.loc[lambda df: df.hue ==2, :]\
                ['Distance to edge of stimulation electrode']]

        plt.hist(x, color = [my_pal[0], my_pal[1], my_pal[2]], histtype= 'barstacked')
        plt.xlabel('Distance to edge\n' r'of stimulation electrode [$\mu m$]')
        plt.ylabel('Number of cells')


        f.add_subplot(2,2,4)
        x = [df.loc[lambda df: df.hue ==0, :]\
                ['Robustness Index'],
            df.loc[lambda df: df.hue ==1, :]\
                ['Robustness Index'],
            df.loc[lambda df: df.hue ==2, :]\
                ['Robustness Index']]
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))
        plt.hist(x, color = [my_pal[0], my_pal[1], my_pal[2]], histtype= 'barstacked')
        plt.xlabel('Reliability Index')
        plt.ylabel('Number of cells')
        for ax in f.get_axes():
            ax.spines["top"].set_visible(False)

            ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fname = 'dist_ri_hists'
        if path is not None:
            plt.savefig(path+fname+'.png', dpi = 300, fmt='png')
    #         plt.savefig(path+fname+'.pdf', dpi = 300, fmt='pdf')
            plt.savefig(path+fname+'.svg', dpi = 300, fmt='svg')


def model_performance_overview(spikes, sta, glm, model, stimulus, cells, \
                               all_cells = None, robust_cells = None, plot_path = None, \
                               clustered = True, start = 0, stop = 0.5):
    
    assert stop>start, 'Your stop value is not larger than your start value'
#     mpl.rcParams.update({'font.size':15})
#     mpl.rc('axes', labelsize = 15)
    plt.style.use('seaborn-paper')
    mpl.rcParams.update({'font.size':13, 'axes.labelsize':13,\
                          'xtick.labelsize':13, 'ytick.labelsize':13})
    f_ = 2
    nrows=f_*len(cells)+4
    ncols=12
    fig = plt.figure(figsize=(12,8))
#     fig = plt.figure()
    show_std = False
    light_blue = sns.color_palette('RdBu', n_colors=10)[-3]
    sta_axes = []
    nonlin_axes = []
    lnp_pred_axes = []
    glm_pred_axes = []
    for i, cell in enumerate(cells):
        '''
        plot the STA
        '''
        print(f_*i)
        ax = plt.subplot2grid([nrows, ncols], loc = [i*f_,0], colspan=2, rowspan=2)
        sta_axes.append(ax)
        x = np.linspace(-0.02,0.01,sta.len_sta_samples)    
              
        sta_white = np.average(sta.full_sta[cell], 0)
        std = np.std(np.concatenate(sta.full_ste[cell], axis=1), axis=1)
        if clustered:
            cl = sta.color_mapping[sta.colors_dict[cell]][0]
            
        else:
            cl = 'k'
        plt.plot(x, sta_white, color = cl, label = 'LNP filter')
        if show_std:
            plt.fill_between(x, sta_white-std, sta_white+std, color = cl, alpha = 0.4)

        (ymin, ymax) = ax.get_ylim()
        (xmin, xmax) = ax.get_xlim()
        plt.vlines(0, -0.2, 0.2, color = 'k', linewidth = 0.7)
        plt.hlines(0, xmin=-0.02, xmax=0.01, linestyles='--', linewidths = 0.5)
        
        '''
        plot the GLM beta
        '''
        ax_ = ax.twinx()

        all_betas = [glm.models[cell][i].coef_ for i in
                     range(glm.n_cross_validation)]
        beta = np.squeeze(np.average(all_betas, axis=0))
        if clustered:
            cl = glm.color_mapping[glm.colors_dict[cell]][0]
        else:
            cl = 'k'
        plt.plot(x, beta, color = cl, linestyle='--', label = 'GLM filter')
        
        ax_.set_axis_off()
#         plt.legend()
        
        '''
        plot the nonlinearity
        '''
        if i>0:
            ax = plt.subplot2grid([nrows, ncols], loc = [i*f_,2], colspan=2, \
                                  sharex = nonlin_axes[i-1], rowspan=2)
#         for n_partition, partition in \
#             enumerate(model.binned_linear_prediction[cell]):

        else: 
            ax = plt.subplot2grid([nrows, ncols], loc = [i*f_,2], colspan=2, rowspan=2)
            
        nonlin_axes.append(ax)
        partition = model.binned_linear_prediction[cell][-1]
        print("partition ", partition)
        n_partition = int(np.floor(start))
        print("n_partition", n_partition)
        y = np.average(model.binned_nonlinear_prediction[cell],axis = 0)
        plt.scatter(partition, \
                    y/4, color = 'k')
        
        r_sig = model.r_sig_all[cell][n_partition]
        r_exp = model.r_exp_all[cell][n_partition]
        
        if r_sig>r_exp:
    
            if model.success_sig_all[cell][n_partition]:
                sig_pred = sigmoid(partition, *model.popt_sig_all[cell][n_partition])
                plt.plot(partition, sig_pred, color = 'k', label = 'r_sig = '+str(r_sig))

        else:
            if model.success_exp_all[cell][n_partition]:
                exp_pred = exponential(partition, *model.popt_exp_all[cell][n_partition])
                print(model.popt_exp_all[cell])
                plt.plot(partition, exp_pred, color = light_blue, label = 'r_exp = '+str(r_exp))
                        
        '''
        plot the true firing rate 
        '''
                        
        ax = plt.subplot2grid([nrows, ncols], loc = [f_*i,4], colspan=4, rowspan=2)
        lnp_pred_axes.append(ax)
        stop = stop
        f = 1000
        bins = np.linspace(start, stop, int(f*stop))
        
        spikes.plot_psth(cell, bins, ax=ax, normed=True, ms=True, \
                         annotate_max = False, alpha = 1)
        
        '''
        plot the LNP prediction
        '''
        x = np.linspace(start, stop, int((f*stop-f*start)))
        if clustered:
            cl_LNP = sta.color_mapping[sta.colors_dict[cell]][0]
            
        else:
            cl_LNP = light_blue

        start_idx = int((f*start)-(f*n_partition))
        stop_idx = int((f*stop)-(f*n_partition))
        ax.plot(x[model.roll_factor:], model.nonlin_prediction[cell][n_partition]\
                    [start_idx+model.roll_factor:stop_idx], color = cl_LNP, alpha = 1, \
                    label = model.nonlin_corr[cell][n_partition])
        
        lnp_pred = np.round(np.average(model.nonlin_corr[cell]), 2)
        ax.annotate(str(lnp_pred), xy = (0.8, 0.8), xycoords='axes fraction')
        (ymin, ymax) = ax.get_ylim()
        ax.set_ylim((0, ymax))
        '''
        plot the true firing rate 
        '''
                        
        ax = plt.subplot2grid([nrows, ncols], loc = [f_*i,8], colspan=4, rowspan=2)
        glm_pred_axes.append(ax)        
        spikes.plot_psth(cell, bins, ax=ax, normed=True, ms=True, \
                         annotate_max = False, alpha = 1)

        '''
        plot the GLM prediction
        '''
        roll_factor = 0.02
        x = np.linspace(start+roll_factor, stop+roll_factor,int((stop-start)*f))
        if clustered:
            cl_GLM = glm.color_mapping[glm.colors_dict[cell]][0]
            
        else:
            cl_GLM = light_blue
            
        ax.plot(x[:int(-roll_factor*f)], \
                glm.nonlin_prediction[cell][n_partition][start_idx:stop_idx-int(f*roll_factor)],
                color = cl_GLM, alpha = 1)
        glm_pred = np.round(np.average(glm.nonlin_corr[cell]), 2)
        ax.annotate(str(glm_pred), xy = (0.8, 0.8), xycoords='axes fraction')
        (ymin, ymax) = ax.get_ylim()
        ax.set_ylim((0, ymax))
        
    for ax in fig.get_axes():
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

    for ax in sta_axes:
        ax.spines["right"].set_visible(False)
        for label in ax.get_yticklabels():
            label.set_visible(False)
            
        ax.tick_params(which = 'both', axis = 'both',left = False)
            
        
    for ax in sta_axes[:-1]:
        ax.tick_params(which = 'both', axis = 'both', bottom = False)
        for label in ax.get_xticklabels():
            label.set_visible(False)
    
    sta_axes[-1].set_xlabel('Time [s]')
    sta_axes[-2].set_ylabel('Filter amplitude [AU]')
            
    for ax in nonlin_axes[:-1]:
        ax.tick_params(which = 'both', axis = 'both', bottom = False)
#         ax.yaxis.tick_right()
        for label in ax.get_xticklabels():
            label.set_visible(False)
            
    for ax in nonlin_axes:
        ax.yaxis.tick_right()
            
        ax.set_xlim((-2.5, 6))
        
    nonlin_axes[-1].set_xlabel('Linear filter \n response [AU]')
#     nonlin_axes[-2].set_ylabel('Predicted firing rate [spikes/ms]')
    nonlin_axes[-2].yaxis.set_label_position('right')
    
    for ax in lnp_pred_axes[:-1]:
        for label in ax.get_xticklabels():
            label.set_visible(False)
            
        ax.tick_params(which = 'both', axis = 'both', bottom = False)
    for ax in lnp_pred_axes:
        for label in ax.get_yticklabels():
            label.set_visible(False)
        ax.tick_params(which = 'both', axis = 'y', bottom = False, left=False)
        
    lnp_pred_axes[-1].set_xlabel('Time [s]')
#     lnp_pred_axes[-2].set_ylabel('(Predicted) firing rate [spikes/ms]')
    for ax in glm_pred_axes[:-1]:
        for label in ax.get_xticklabels():
            label.set_visible(False)
            
        ax.tick_params(which = 'both', axis = 'both', bottom = False)
        
    for ax in glm_pred_axes:
#         for label in ax.get_yticklabels():
#             label.set_visible(False)
        ax.yaxis.tick_right()
        ax.tick_params(which = 'both', axis = 'both', bottom = False, left=False)
     
    glm_pred_axes[-1].set_xlabel('Time [s]')       
    glm_pred_axes[-2].set_ylabel('(Predicted) firing rate [spikes/ms]')
    glm_pred_axes[-2].yaxis.set_label_position('right')

    '''
    ###################################################################################################################
    overview plots
    '''
    if all_cells is not None:
        summary_plots = []
        plt.subplot2grid([nrows+3, ncols+3], loc = [9,1], colspan=4, rowspan=4)
        x = [np.average(model.nonlin_corr[cell]) for cell in all_cells]
        y = [np.average(glm.nonlin_corr[cell]) for cell in all_cells]
        
        if clustered:
            cluster_label = [sta.cluster_labels.get(cell, 3) for cell in all_cells]
            style = ['o' if label in [0, 1, 2] else '.' for label in cluster_label]
            # d = {'STA fit performance': x, 'MLE fit performance': y, 'cluster label':
            #     [sta.cluster_labels[cell] for cell in robust_cells]}
            d = {'STA fit performance': x,
                 'MLE fit performance': y,
                 'cluster label': cluster_label}
            style_order = ['o', '.']
        else:
            style = ['.' if cell in robust_cells else 'o' for cell in all_cells]
            cluster_label = [3 if cell in robust_cells else 4 for cell in all_cells]
            d = {'STA fit performance': x, 'MLE fit performance': y, 'cluster label':
                cluster_label}

            style_order = ['.', 'o']

        df = pd.DataFrame(d)
        dark_green = sns.color_palette('PiYG', n_colors=10)[-1]
        dark_red = sns.color_palette('YlOrRd', n_colors=10)[-1]
        dark_blue = sns.color_palette('RdBu', n_colors=10)[-1]
        dark_purple =  sns.color_palette('PuOr', n_colors=10)[-1]
        my_pal = {0 : dark_green, 1:dark_red, 2:dark_purple, 3:'k', 4: 'k', -1:'k'}
        ax = sns.scatterplot(x = 'STA fit performance', y = 'MLE fit performance',
                             hue = 'cluster label', data = df, palette = my_pal,
                             style=style, markers=['o', '.'], style_order=style_order,
                             legend = False)

        lnp_robust_perf = np.asarray([np.average(model.nonlin_corr[cell]) for cell in robust_cells])
        glm_robust_perf = np.asarray([np.average(glm.nonlin_corr[cell]) for cell in robust_cells])
        print(
            'LNP : mean:{:.3g}, std : {:.3g}'.format(lnp_robust_perf.mean(), lnp_robust_perf.std()))
        print(
            'GLM : mean:{:.3g}, std : {:.3g}'.format(glm_robust_perf.mean(), glm_robust_perf.std()))
        print(
            'LNP : min:{:.3g}, max : {:.3g}'.format(lnp_robust_perf.min(), lnp_robust_perf.max()))
        print(
            'GLM : min:{:.3g}, max : {:.3g}'.format(glm_robust_perf.min(), glm_robust_perf.max()))
        for i in set(cluster_label):
            x_mean = df['STA fit performance'].loc[df['cluster label']==i].mean()
            y_mean = df['MLE fit performance'].loc[df['cluster label']==i].mean()
            x_std = df['STA fit performance'].loc[df['cluster label']==i].std()
            y_std = df['MLE fit performance'].loc[df['cluster label']==i].std()
            ax.errorbar(x_mean, y_mean, x_std, y_std, color = my_pal[i])
            print('LNP cluster : {:.3g}, mean:{:.3g}, std : {:.3g}'.format(i, x_mean, x_std))
            print('GLM cluster : {:.3g}, mean:{:.3g}, std : {:.3g}'.format(i, y_mean, y_std))
            lnpmin = df['STA fit performance'].loc[df['cluster label']==i].min()
            lnpmax = df['STA fit performance'].loc[df['cluster label']==i].max()
            print('LNP cluster:{:.3g}, min:{:.3g}, max:{:.3g}'.format(i, lnpmin, lnpmax))
            glmmin = df['MLE fit performance'].loc[df['cluster label'] == i].min()
            glmmax = df['MLE fit performance'].loc[df['cluster label'] == i].max()
            print('GLM cluster :{:.3g}, min:{:.3g}, max:{:.3g}'.format(i, glmmin, glmmax))

        ax.set_aspect('equal', adjustable='box')
        summary_plots.append(ax)
        '''
        ######################################################################
        reliability vs STA fit performance
        '''
        x = [np.average(model.nonlin_corr[cell]) for cell in all_cells]
        lin_perf = [np.average(model.lin_corr[cell]) for cell in all_cells]
        plt.subplot2grid([nrows+3, ncols+3], loc = [9,6], colspan=4, rowspan=4,
                         sharex = summary_plots[0], sharey = summary_plots[0])
        benchmark = [np.average(spikes.benchmark[cell]) for cell in all_cells]
        
        d1 = {'Reliability Index' : benchmark, 'STA fit performance' : x,
              'Linear stage performance' : lin_perf, 'cluster label' : cluster_label}
        df1 = pd.DataFrame(d1)
        
        ax = sns.scatterplot(x = 'Reliability Index', y = 'STA fit performance',
                             hue='cluster label',data=df1, palette=my_pal,
                             style=style, markers=['o', '.'],
                             style_order=style_order,
                             legend = False)

        print('LNP mean and std : {},  {}'.format(df1['STA fit performance'].mean(),\
                                                  df1['STA fit performance'].std()))
        r, p = pearsonr(df1['Reliability Index'], df1['Linear stage performance'])
        # print('Corr RI Linear stage; r:{:.3g}, p:{:.3g}'.format(r, p))
        # if clustered:
        for i in set(cluster_label):
            x_mean = df1['Reliability Index'].loc[df1['cluster label']==i].mean()
            y_mean = df1['STA fit performance'].loc[df1['cluster label']==i].mean()
            x_std = df1['Reliability Index'].loc[df1['cluster label']==i].std()
            y_std = df1['STA fit performance'].loc[df1['cluster label']==i].std()
            # print('LNP mean perf; Cluster :{:.3g}, mean:{:.3g}, std:{:.3g}'.format(i, y_mean, y_std))
            ax.errorbar(x_mean, y_mean, y_std, x_std, color = my_pal[i])
            x_temp = df1['Reliability Index'].loc[df1['cluster label']==i]
            y_temp = df1['STA fit performance'].loc[df1['cluster label']==i]
            r, p = pearsonr(x_temp,y_temp)
            print('LNP corr reliability perf; Cluster :{:.3g}, r:{:.3g}, p:{:.3g}'.format(i, r, p))

        ax.set_aspect('equal', adjustable='box')
        summary_plots.append(ax)

        '''
        ######################################################################
        perf lin stage vs perf full model
        '''
        
        lin_perf = [np.average(model.lin_corr[cell]) for cell in all_cells]
           
        plt.subplot2grid([nrows+3, ncols+3], loc = [9,11], colspan=4, rowspan=4,
                         sharex = summary_plots[0], sharey = summary_plots[0])
        d1 = {'Linear stage performance' : lin_perf, 'STA fit performance' : x, \
              'cluster label' : cluster_label}
        df1 = pd.DataFrame(d1)


        ax = sns.scatterplot(x = 'Linear stage performance',
                             y = 'STA fit performance',
                             hue = 'cluster label',
                             data=df1,
                             palette=my_pal,
                             style=style,
                             markers=['o', '.'],
                             style_order=style_order,
                             legend = False)

        for i in set(cluster_label):
            x_mean = df1['Linear stage performance'].loc[df1['cluster label']==i].mean()
            y_mean = df1['STA fit performance'].loc[df1['cluster label']==i].mean()
            x_std = df1['Linear stage performance'].loc[df1['cluster label']==i].std()
            y_std = df1['STA fit performance'].loc[df1['cluster label']==i].std()
#                 print('LNP; Cluster :{:.3g}, mean:{:.3g}, std:{:.3g}'.format(i, y_mean, y_std))
            ax.errorbar(x_mean, y_mean, y_std, x_std, color = my_pal[i])
            x_temp = df1['Linear stage performance'].loc[df1['cluster label']==i]
            y_temp = df1['STA fit performance'].loc[df1['cluster label']==i]
            r, p = pearsonr(x_temp,y_temp)
            print('LNP corr lin nonlin; Cluster :{:.3g}, r:{:.3g}, p:{:.3g}'.format(i, r, p))
            st, p_value = wilcoxon(x_temp, y_temp)
            print('Wilcoxon test linear nonlinear stage; Cluster :{:.3g}, stats:{:.3g}, p:{:.3g}'.format(i, st, p_value))

        ax.set_aspect('equal', adjustable='box')
        summary_plots.append(ax)
        for ax in summary_plots:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            (xmin, xmax) = ax.get_xlim()
            (ymin, ymax) = ax.get_ylim() 
            abs_min = min([xmin, ymin])
            abs_max = max([xmax, ymax])
            ax.set_xlim((abs_min, 0.75))
            ax.set_ylim((abs_min, 0.75))
            ax.plot([abs_min, abs_max], [abs_min, abs_max], color = 'k')
            # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        
    mpl.rcParams.update({'font.size':15})
    plt.subplots_adjust(wspace = 2, hspace = 1)
    if plot_path is not None:
        plt.savefig(plot_path+'paper_model_fig.png', fmt = 'png', dpi = 300)
        plt.savefig(plot_path+'paper_model_fig.pdf', fmt = 'pdf', dpi = 300)
        plt.savefig(plot_path+'paper_model_fig.svg', fmt = 'svg', dpi = 300)

    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    pass
