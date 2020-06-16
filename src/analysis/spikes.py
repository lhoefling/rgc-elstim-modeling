'''
Created on 27.07.2018

@author: larissa.hoefling@uni-tuebingen.de
'''
from copy import deepcopy

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


class Spikes():
    '''
    ###########################################################################
    To do list:
        init needs a file dialog
        structure_timestamps needs to read the number of appended files (= # of 
        repetitions) automatically
    
    ###########################################################################
    Objects of the class spikes contain the raw spike times of all cells in one
    measurement. They have the following instance variables and methods defined 
    on them:
    
    variables:
    ###########################################################################
    path:        string
                 path to the .cmtr file; can be hardcoded or prompted with a GUI
    stim_dur:     int
                  duration of 1 stimulus repetition in ms
    n_stim_reps:    int
                    number of stimulus repetitions
                    
    
    
    ts:          pandas dataframe
                 see description of read_spikes()
        
    
    structured_ts:    dictionary
                      see description of structure_timestamps()
    robust_spikes:    dictionary (same structure as structured_ts)
                      containing only roubst spikes
    partition:        list of lists
                      the partition used for creating training and test set
    training_set, test_set, robust_training_set, robust_test_set:     dictionary
    
    
    methods:
    ###########################################################################
    read_spikes()
    returns a data frame containing the time stamps of all cells. The data frame
    contains n rows (n = number of spike times of all cells) and 5 columns;
    index, unit, time_stamps, boss_column, boss_row, tags. This method is called 
    upon initialization, and the output is stored in the variable ts
    
    structure_timestamps()
    
    
    '''

    def __init__(self, path=None, cells=None, isicheck=False, limit=0.02, \
                 tag_check=False, binwidth=2, lim=3, stim_els=None):
        '''
        set the path to the corresponding .cmtr (h5) file
        '''
        if path == None:
            # fill this later with a gui
            pass
        else:
            self.path = path
        self.joined = False

        self.stim_dur = 5000  # duration of 1 stimulus repetition in ms
        self.n_stim_reps = 5  # this should be read automatically from file
        self.read_spikes(isicheck, limit, tag_check)
        self.structure_timestamps(cells)
        #         self.dummy_spikes = self.create_dummy_spikes(cells=cells)
        self.create_robust_spikes(binwidth, lim, cells=cells)
        self.partition = [[0, 1000], [1000, 2000], [2000, 3000], [3000, 4000], \
                          [4000, 5000]]

        training_set, test_set = \
            self.create_training_test_set(partition=self.partition, cells=cells)
        self.training_set = training_set
        self.test_set = test_set
        self.robust_training_set, self.robust_test_set = \
            self.create_training_test_set(partition=self.partition, cells=cells, \
                                          robust=True)
        self.compute_benchmark()
        if stim_els is not None:
            cells = [cell for cell in self.structured_ts.keys()]
            self.find_dist_to_els(stim_els, cells)

    def read_spikes(self, isicheck=False, limit=0.02, tag_check=False):
        '''
        reads the time stamps, cell position and (possibly) tags of all cells 
        from one recording and stores them in a pandas dataframe
        with columns unit (unit ID), time_stamps (in ms), boss_column (x position)
        boss_row (y position) 
        
        Inputs: 
        -----------------------------------------------------------------------
        isicheck:     Boolean
                      determines whether the percentage of ISIs < 1 ms should be
                      checked for
        limit:        float
                      in case if isicheck, determines the maximal tolerated 
                      proportion of ISIs < 1 ms
                      
        Outputs:
        -----------------------------------------------------------------------
        None;
        sets self.ts as the dataframe 
        '''

        factor = 0.001  # factor to convert from us to ms
        units = []
        time_stamps_all = []
        boss_column = []
        boss_row = []
        tags = []
        with h5py.File(self.path, 'r') as f:
            spike_sorter_output = f['Spike Sorter']
            iterator = spike_sorter_output.__iter__()
            for name in iterator:

                if name.find('Unit ') >= 0:
                    temp = spike_sorter_output.get(name)
                    unitInfo = temp.get('Unit Info')
                    tag = unitInfo['Tags']
                    if tag_check:
                        if tag == b'':

                            peaks = temp.get('Peaks')
                            mask = np.array(peaks['IncludePeak'], dtype=bool)
                            time_stamps = peaks['Timestamp'][mask]
                            time_stamps = time_stamps * factor
                            num = len(time_stamps)
                            unitNo = int(name.split(' ')[-1])
                            if isicheck:

                                res = self.check_isi(time_stamps, limit)
                                if res:

                                    units.append(np.repeat(unitNo, num))
                                    time_stamps_all.append(time_stamps)
                                    col = temp.get('Unit Info')['Column']
                                    row = temp.get('Unit Info')['Row']
                                    boss_column.append(np.repeat(col, num))
                                    boss_row.append(np.repeat(row, num))
                                    if tag:
                                        tags.append(np.repeat(tag, num))
                                    else:
                                        tag = 'None'
                                        tags.append(np.repeat(tag, num))
                            else:
                                units.append(np.repeat(unitNo, num))
                                time_stamps_all.append(time_stamps)
                                col = temp.get('Unit Info')['Column']
                                row = temp.get('Unit Info')['Row']
                                boss_column.append(np.repeat(col, num))
                                boss_row.append(np.repeat(row, num))
                                if tag:
                                    tags.append(np.repeat(tag, num))
                                else:
                                    tag = 'None'
                                    tags.append(np.repeat(tag, num))
                        else:
                            print(self.path)
                            print(name, temp.get('Unit Info')['Column'], \
                                  temp.get('Unit Info')['Row'])
                    else:
                        peaks = temp.get('Peaks')
                        mask = np.array(peaks['IncludePeak'], dtype=bool)
                        time_stamps = peaks['Timestamp'][mask]
                        time_stamps = time_stamps * factor
                        num = len(time_stamps)
                        unitNo = int(name.split(' ')[-1])
                        if isicheck:

                            res = self.check_isi(time_stamps, limit)
                            if res:

                                units.append(np.repeat(unitNo, num))
                                time_stamps_all.append(time_stamps)
                                col = temp.get('Unit Info')['Column']
                                row = temp.get('Unit Info')['Row']
                                boss_column.append(np.repeat(col, num))
                                boss_row.append(np.repeat(row, num))
                                if tag:
                                    tags.append(np.repeat(tag, num))
                                else:
                                    tag = 'None'
                                    tags.append(np.repeat(tag, num))
                        else:
                            units.append(np.repeat(unitNo, num))
                            time_stamps_all.append(time_stamps)
                            col = temp.get('Unit Info')['Column']
                            row = temp.get('Unit Info')['Row']
                            boss_column.append(np.repeat(col, num))
                            boss_row.append(np.repeat(row, num))
                            if tag:
                                tags.append(np.repeat(tag, num))
                            else:
                                tag = 'None'
                                tags.append(np.repeat(tag, num))

        d = {'unit': pd.Series(np.concatenate(units)), \
             'time_stamps': pd.Series(np.concatenate(time_stamps_all)), \
             'boss_column': pd.Series(np.concatenate(boss_column)), \
             'boss_row': pd.Series(np.concatenate(boss_row)), \
             'tags': pd.Series(np.concatenate(tags))}

        df = pd.DataFrame(d)
        self.ts = df

    def structure_timestamps(self, cells=None):

        '''
        input:
        -----------------------------------------------------------------------
        
        cells         if cells=None, return the structured timestamps for
                      all cells in the recording. if cells is a list of integers,
                      return the structured timestamps for the specified cells 
                      (index into the column "units" in self.ts, starting at 1)
                    
        
        
        output: 
        -----------------------------------------------------------------------
        None; set self.structured_ts to
        time_stamps_all    a dictionary with cells as keys and
            time_stamps    a list of length n_stim_reps, each element containing
                           an array with that cell's spikes in that repetition
                            as values
                    sets self.cell_to_loc to
                    a dictionary with cell IDs as keys and a tuple of (x_position,
                    y_position) of the cells as values. x and y position are 
                    given in recording electrode coordinates 
        
        
        '''
        n_stim_reps = self.n_stim_reps  # this should be read automatically from the file
        stim_start = 0  # this should be read automatically from the file
        stim_dur = 5000  # this should be read automatically from the file

        if cells == None:
            cells = list(set(self.ts['unit']))

        time_stamps_all = {cell: [] for cell in cells}
        cell_to_loc = {cell: [] for cell in cells}
        for cell in cells:
            temp = np.array(self.ts.loc[self.ts['unit'] == cell, 'time_stamps'])

            cell_row = np.array(self.ts.loc[self.ts['unit'] == cell, 'boss_row'])[0]
            cell_column = np.array(self.ts.loc[self.ts['unit'] == cell, 'boss_column'])[0]
            cell_to_loc[cell] = (cell_column, cell_row)

            time_stamps = []
            for j in np.arange(0, n_stim_reps):
                condition1 = temp >= (j + 1) * stim_start + j * stim_dur
                condition2 = temp <= (j + 1) * stim_start + (j + 1) * stim_dur
                norm_spikes = temp[condition1 & condition2] - ((j + 1) * stim_start + j * stim_dur)
                if len(norm_spikes) <= 1:
                    time_stamps.append(np.zeros(1))
                else:
                    time_stamps.append(norm_spikes)

            time_stamps_all[cell] = time_stamps

        self.cell_to_loc = cell_to_loc
        self.structured_ts = time_stamps_all

    def find_dist_to_els(self, stim_els, cells):

        self.dist_to_stim = {cell: None for cell in cells}
        self.closest_stim_el = {cell: None for cell in cells}
        for cell in cells:
            min_dist, min_tup = self.min_tuple_dist(stim_els, \
                                                    self.cell_to_loc[cell])
            self.dist_to_stim[cell] = min_dist
            self.closest_stim_el[cell] = min_tup

    def create_training_test_set(self, partition, cells=None, robust=False):
        '''
        This method creates a training and a test set out of the given spikes 
        for cells. 
        Input: 
        -----------------------------------------------------------------------
        partition    a list of lists 
                    gives the start and end points of the excerpts
                    from the recordings which shall constitute the training and 
                    test data set. E.g. [[0, 1000], [1000,2000], ... ,[4000,5000]]
                    will create 5 training sets containing 4 s (4000ms)
                    of the recording and 5 corresponding test sets 
                    containing the remaining 1 s [1000ms] of the recording
                    
        output:       
        ----------------------------------------------------------------------- 
        training_time_stamps_all, test_time_stamps_all
                    a dictionary with cell identities as keys, and lists as 
                    values. Each list has n_partition elements, each element
                    containing n_rep lists that contain the spikes of one
                    stimulus repetition that belong into the corresponding
                    partition
        
        '''
        if cells == None:
            cells = list(set(self.ts['unit']))

        training_time_stamps_all = {cell: [] for cell in cells}
        test_time_stamps_all = {cell: [] for cell in cells}
        for cell in cells:
            if robust:
                time_stamps = self.robust_spikes[cell]
            else:
                time_stamps = self.structured_ts[cell]
            training_time_stamps = [[] for _ in time_stamps]
            test_time_stamps = [[] for _ in time_stamps]

            for n_partition, element in enumerate(partition):
                for spikes in time_stamps:
                    cond1 = spikes > element[0]
                    cond2 = spikes <= element[1]
                    cond = np.logical_and(cond1, cond2)
                    test_time_stamps[n_partition].append(spikes[cond])
                    not_cond = np.logical_not(cond)
                    training_time_stamps[n_partition].append(spikes[not_cond])

            training_time_stamps_all[cell] = training_time_stamps
            test_time_stamps_all[cell] = test_time_stamps

        return training_time_stamps_all, test_time_stamps_all

    def create_spike_histogram(self, lower_limit=20, upper_limit=0, \
                               binsize=1, mode='full', set_no=None):
        '''
        operates on all cells
        creates a spike histogram with defined binsize out of the spike times
        
        Inputs:
        -----------------------------------------------------------------------
        lower_limit:    int
                        include spikes occurring at t>lower_limit
        upper_limit     int
                        include spikes occurring at t<upper_limit
        binsize         int
                        bin size for the histogram
                        
        Outputs:
        -----------------------------------------------------------------------
        hist_all:        a dictionary with cell IDs as keys and lists of lists 
                         as values, where hist_all[cell_ID][n] contains the 
                         histogram of spikes of cell cell_ID in repetition n
                         

                        
        '''
        if mode == 'train' or mode == 'test':
            assert set_no is not None, "You passed train or test as mode, but did\
            not provide a valid partition number"
        assert self.stim_dur % binsize == 0, "Stimulus duration and chosen binsize\
        are not compatible"
        n_bins = self.stim_dur // binsize
        if not (lower_limit % binsize == 0) or not (upper_limit % binsize == 0) or \
                not (self.stim_dur % binsize == 0):
            raise ValueError('The chosen input values are incompatible')

        cells = self.structured_ts.keys()
        hist_all = {cell: [[] for n_rep in range(len(self.structured_ts[cell]))] \
                    for cell in cells}
        if mode == 'full':

            for cell in cells:
                for n_rep, spikes in enumerate(self.structured_ts[cell]):
                    '''round to ms precision '''
                    rounded_spikes = np.ceil(spikes).astype(int) - 1
                    temp_hist_raw = np.zeros(self.stim_dur)
                    temp_hist_raw[rounded_spikes] = 1
                    if binsize > 1:
                        temp_hist = np.reshape(temp_hist_raw, (-1, binsize))
                        temp_hist = np.sum(temp_hist, axis=1, dtype=int)
                    else:
                        temp_hist = temp_hist_raw
                    hist_all[cell][n_rep] = \
                        temp_hist[int(lower_limit / binsize):len(temp_hist_raw) + \
                                                             int(upper_limit / binsize)]

        elif mode == 'train':
            print("histogram train")
            for cell in cells:
                for n_rep, spikes in enumerate(self.training_set[cell][set_no]):
                    '''round to ms precision '''
                    p = self.partition[set_no]
                    rounded_spikes = np.ceil(spikes).astype(int) - 1
                    temp_hist_raw = np.zeros(n_bins)
                    temp_hist_raw[rounded_spikes] = 1
                    temp1 = temp_hist_raw[lower_limit:p[0] + lower_limit]
                    temp2 = temp_hist_raw[p[1] + lower_limit:self.stim_dur + upper_limit]
                    #                     print("First snippet:{} {}, second snippet: {} {}\
                    #                     ".format(lower_limit, p[0]+lower_limit, p[1]+lower_limit, \
                    #                              self.stim_dur+upper_limit))
                    #
                    temp_hist_raw = np.concatenate([temp1, temp2])
                    if binsize > 1:
                        temp_hist = np.reshape(temp_hist_raw, (-1, binsize))
                        temp_hist = np.sum(temp_hist, axis=1, dtype=int)
                    else:
                        temp_hist = temp_hist_raw
                    hist_all[cell][n_rep] = temp_hist



        elif mode == 'test':
            print("histogram test")
            for cell in cells:
                for n_rep, spikes in enumerate(self.test_set[cell][set_no]):
                    '''round to ms precision '''
                    p = self.partition[set_no]
                    rounded_spikes = np.ceil(spikes).astype(int) - 1
                    temp_hist_raw = np.zeros(n_bins)
                    temp_hist_raw[rounded_spikes] = 1
                    start = p[0] + lower_limit
                    stop = p[1] + lower_limit
                    if stop > self.stim_dur:
                        stop = stop - lower_limit - upper_limit
                    temp_hist_raw = temp_hist_raw[start:stop]
                    if binsize > 1:
                        temp_hist = np.reshape(temp_hist_raw, (-1, binsize))
                        temp_hist = np.sum(temp_hist, axis=1, dtype=int)
                    else:
                        temp_hist = temp_hist_raw
                    hist_all[cell][n_rep] = temp_hist


        else:
            raise ValueError("invalid mode")

        return hist_all

    def create_robust_spikes(self, binwidth=2, lim=2, cells=None):
        '''
        creates a variable self.robust_spikes, structured like structured_ts, 
        but keeping only those spikes that occur more than lim times within the
        same binwidth-wide time window across all repetitions
        
        Inputs:
        binwidth    int
                    width of the time window in ms
        lim         int
                    determines the minimum number of spikes that have to occur
                    in the specified time window to be retained
                    
        '''
        self.robust_cells = []

        if cells == None:
            cells = list(set(self.ts['unit']))
        self.stim_dur = 5000
        n_bins = int(self.stim_dur / binwidth) + 2
        bin_edges = np.linspace(0, self.stim_dur + binwidth, n_bins)
        n_rep = self.n_stim_reps
        self.robust_spikes = {cell: [[] for _ in range(n_rep)] for cell in \
                              self.structured_ts.keys()}
        self.robustness_ratio = {cell: [[] for _ in range(n_rep)] for cell in \
                                 self.structured_ts.keys()}
        for cell in cells:
            spikes = np.concatenate(self.structured_ts[cell])

            hist_values, _ = np.histogram(spikes, bins=bin_edges)
            digital = np.digitize(spikes, bin_edges, right=False) - 1
            lower = 0
            for n, rep in enumerate(self.structured_ts[cell]):
                upper = lower + len(rep)
                self.robust_spikes[cell][n] = \
                    rep[hist_values[digital[lower:upper]] > lim]

                lower = upper
            n_rob_spikes = len(np.concatenate(self.robust_spikes[cell]))
            n_spikes = len(spikes)
            self.robustness_ratio[cell] = n_rob_spikes / n_spikes

            assert np.sum(hist_values[hist_values > lim]) == \
                   len(np.concatenate(self.robust_spikes[cell])), \
                'There is something wrong in create_robust_spikes(), cell{}'.format(cell)

    #             self.

    def check_isi(self, time_stamps, limit):
        '''
        Checks whether the percentage of ISIs (diffs) in time_stamps exceeds a 
        proportion of limit
        Inputs:
        -----------------------------------------------------------------------
        time_stamps:    list of timestamps
        limit:          float
        '''
        isi = np.diff(time_stamps)

        percentage_smaller = len(np.where(isi < 1)[0]) / len(isi)

        return percentage_smaller < limit

    def raster(self, cell, ax=None, mode='full', ms=False, stop=1, \
               robust=False, density=False, raster_show=True, \
               cmap='Greys', sigma=15, **kwargs):
        """
        Creates a raster plot
    
        Inputs:
        ----------
        cell    cell ID (used as key into self.structured_ts or self.robust_spikes)
        ax      matplotlib.pyplot axes object
                if passed, plot into these axes
        mode:     string
                deprecated
        ms      boolean
                if True, convert spikes to s
        stop    int
                x-axis limit
        robust    boolean
                  if True use self.robust_spikes instead of self.structured_ts
        density    boolean    
                    if True, display a spike density estimate on top of the raster
        raster_show    boolean
                       display raster or not
        cmap        string
                    color map
        
    
        outputs
        -------
        none
        """
        if ax is not None:
            plt.sca(ax)

        if mode == 'full':
            if robust:
                event_times_list = self.robust_spikes[cell]

            else:
                event_times_list = self.structured_ts[cell]
            #         elif mode == 'tnt'

            if raster_show:
                for ith, trial in enumerate(event_times_list):
                    if ms:
                        # divide by 1000 to convert to ms
                        trial = trial / 1000
                        plt.vlines(trial, ith + .5, ith + 1.5, linewidths=0.5, \
                                   **kwargs)
                    else:

                        plt.vlines(trial, ith + .5, ith + 1.5, linewidths=0.5, \
                                   **kwargs)

            plt.xlim(0, stop)
            plt.ylim(.5, len(event_times_list) + .5)

            if density:
                spike_density = np.concatenate(event_times_list)
                kdefactor = sigma / np.sqrt(np.cov(spike_density, rowvar=1, bias=False))
                g_kde = gaussian_kde(spike_density, bw_method=kdefactor)
                #                 g_kde = gaussian_kde(spike_density, bw_method = 0.025)
                #                 print('The sd of the gaussian kernel for el spike density\
                #                 estimation is {} ms'.format(np.sqrt(g_kde.covariance)))
                (x_min, x_max) = plt.gca().get_xlim()
                (y_min, y_max) = plt.gca().get_ylim()
                points = np.linspace(x_min, x_max * 1000, x_max * 1000)
                density_estimate = g_kde.evaluate(points)
                surf = np.tile(density_estimate, (len(event_times_list), 1))
                plt.imshow(surf, aspect='auto', cmap=cmap, \
                           extent=[x_min, x_max, y_min, y_max])



        elif mode == 'tnt':
            if robust:
                event_times_list_train = self.robust_training_set[cell]
                event_times_list_test = self.robust_test_set[cell]
            else:
                event_times_list_train = self.training_set[cell]
                event_times_list_test = self.test_set[cell]

            for ith, partition in enumerate(event_times_list_train):
                plt.figure()
                for jth, trial in enumerate(partition):
                    plt.vlines(trial, jth + .5, jth + 1.5, color='k', **kwargs)
                    plt.vlines(event_times_list_test[ith][jth], jth + .5, \
                               jth + 1.5, color='r', **kwargs)

    def plot_psth(self, cell, bins, ax=None, normed=False, \
                  ms=False, annotate_max=True, **kwargs):

        '''
        plot the psth of a cell
        
        '''

        if ax is not None:
            plt.sca(ax)

        spikes = np.concatenate(self.structured_ts[cell])

        if ms:
            spikes = spikes / 1000.

        if normed:
            real_rate, bin_edges = np.histogram(spikes, bins)
            normed_real_rate = real_rate / self.n_stim_reps
            width = bin_edges[1] - bin_edges[0]
            plt.bar(bin_edges[:-1], normed_real_rate, width=width, \
                    align='edge', color='gray', edgecolor='k', **kwargs)

            y_max = np.max(normed_real_rate)
            if annotate_max:
                plt.annotate(str(y_max), (0, y_max))

            return normed_real_rate

        else:
            plt.hist(spikes, bins=bins, color='k', **kwargs)
            real_rate, _ = np.histogram(spikes, bins=bins)

            y_max = np.max(real_rate)
            if annotate_max:
                plt.annotate(str(y_max), (0, y_max))
        return real_rate

    def compute_benchmark(self, bin_width=2):
        """
        computes the correlation between the spike histogram of n-1 repetitions
        and the histogram of the nth repetition, as a measure of how predictable
        the cell is
        
        Inputs:
        -----------------------------------------------------------------------
        bin_width:    int
                      bin width for the spike histograms
        Outpus:
        -----------------------------------------------------------------------
        None; sets self.benchmark to a dictionary with cell IDs as keys and
                lists as values, each list containing the correlation coefficient
                between all possible n-1 psth/nth psth combinations
        """
        cells = list(self.structured_ts.keys())
        benchmark = {cell: [[] for rep in range(self.n_stim_reps)] for cell in cells}
        bins = np.linspace(0, int(self.stim_dur), int(self.stim_dur / bin_width + 1))
        for cell in cells:
            for i, rep in enumerate(self.structured_ts[cell]):
                predictors = deepcopy(self.structured_ts[cell])
                predictors.pop(i)
                predictors = np.concatenate(predictors)
                predictors_hist, _ = np.histogram(predictors, bins=bins)
                predictors_hist = predictors_hist / (self.n_stim_reps - 1)
                to_predict, _ = np.histogram(rep, bins=bins)
                benchmark[cell][i] = np.round(np.corrcoef(predictors_hist, \
                                                          to_predict)[0, 1], 2)
        self.benchmark = benchmark

    def min_tuple_dist(self, tups, loc):
        min_dist = np.inf
        min_tup = (np.inf, np.inf)
        for tup in range(len(tups)):
            new_dist, _ = self.tuple_distance(tups[tup], loc)
            new_tup = tups[tup]
            idx = np.argmin([min_dist, new_dist])
            min_dist = np.min([min_dist, new_dist])
            min_tup = [min_tup, new_tup][idx]

        return min_dist, min_tup

    def tuple_distance(self, tup1, tup2):
        pitch = 16
        x_dist = np.abs(tup1[0] - tup2[0])
        y_dist = np.abs(tup1[1] - tup2[1])
        euclid_dist = np.sqrt(x_dist ** 2 + y_dist ** 2) * pitch
        dist = np.abs(tup1[0] - tup2[0]) + np.abs(tup1[1] - tup2[1])
        return euclid_dist, dist

    def compute_isi(self, cells):
        self.isi = {cell: np.diff(np.concatenate(self.structured_ts[cell]))
                    for cell in cells}
