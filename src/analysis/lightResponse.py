'''
Created on 28.08.2018

@author: larissa.hoefling@uni-tuebingen.de
'''

from analysis.spikes import Spikes
import h5py
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde




class LightResponse(Spikes):
    '''
    This class inherits from class spikes and reads light responses from file, saving
    them as self.ts using the read_spikes() method from the parent class. The 
    structured time_stamps are saved in a dictionary with keys 'spikes_off' and
    'spikes_on'
    ###########################################################################
    attributes:
    
    stim_dur        int
                    stimulus duration 
    
    path            spike sorted file path
    ts                see spikes
    structured_ts    dictionary with cells as keys and a dict as value; nested
                    dict has spikes_on and spikes_off as values
    on_triggers:    the trigger times, read from file, in s
    off_triggers
    bias_index:     dictionary with cells as keys and bias index as values
    cell_to_loc:    dictionary with cells as keys and cell_column and cell_row
                    as values
                    
                    
    ###########################################################################
    methods:
    
    read_spikes() inherited from spikes
    structure_timestamps overwriting the method from spikes
    raster() overwriting the method from spikes 
    
    
    '''


    def __init__(self, path= None, raw_file_path = None, cells=None):
        '''
        Constructor
        '''
        if path == None:
            pass
        else:
            self.path = path
            
        if raw_file_path is None:
            sep = '/'
            path_split = path.split(sep)
            raw_file = path_split[-1]
            raw_file = raw_file[:-4]+'cmcr'
            path = path_split[:-2]
            path.append(raw_file)
            raw_file_path = sep.join(path)
        
        self.stim_dur = 500
         
        if raw_file_path.find('2017.11.03')>=0:
            with h5py.File(raw_file_path, 'r') as f:
                acq = f['Acquisition']
                EventTool = acq['EventTool @ Digital Data']
                eventData = EventTool['EventData']
                eventID = eventData['EventID']
                trigger_times = eventData['TimeStamp']
                on = trigger_times[eventID==1]
                off = trigger_times[eventID==2]
        elif raw_file_path.find('2018.10.30')>=0:
            on = \
            np.asarray([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,10000 ])
            off = np.asarray([500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500])
            on*=10**3
            off*=10**3
        else: 
            with h5py.File(raw_file_path, 'r') as f:
                acq = f['Acquisition']
                digData = acq['Digital Data Events']
                eventData = digData['EventData']
                eventID = eventData['EventID']
                trigger_times = eventData['TimeStamp']
                on = trigger_times[eventID == 1]
                off = trigger_times[eventID == 2] 
                
      
                 
                 
        self.on_triggers = on/1000
        self.off_triggers = off/1000
        self.read_spikes()
        self.structure_timestamps(cells)
        
        
    def structure_timestamps(self, cells=None):
        '''
        structures the timestamps and saves on and off spikes separately
        
        Output:
        None; sets self.structured_ts to time_stamps_all, a dictionary with 
        cell IDs as keys and as values a nested dictionary with keys spikes_on and 
        spikes_off for every cell
        sets self.cell_to_loc as in spikes.py
        sets self.bias_index to a dictionary with cell IDs as keys and the 
        corresponding bias index as value 
        '''
        
        if cells == None:
            cells = list(set(self.ts['unit']))

        
        time_stamps_all = {cell : {'spikes_on' : [], 'spikes_off' : []} \
                           for cell in cells}
        cell_to_loc = {cell : [] for cell in cells}
        bias_index = {cell : [] for cell in cells}
        transience_index = {cell : [] for cell in cells}
        for cell in cells:
            off_count = 0
            on_count = 0
            spikes_cell = np.array(self.ts.loc[self.ts['unit']==cell, 'time_stamps'])
            cell_row = np.array(self.ts.loc[self.ts['unit']==cell, 'boss_row'])[0]
            cell_column = np.array(self.ts.loc[self.ts['unit']==cell, 'boss_column'])[0]
            cell_to_loc[cell] = (cell_column, cell_row)
            
            off_threshold = 300
            on_threshold = 800
            
            if self.off_triggers[0]<self.on_triggers[0]: 
                for i in range(len(self.off_triggers)):
                    
                    cond1 = spikes_cell>self.off_triggers[i]
                    cond2 = spikes_cell<self.off_triggers[i]+self.stim_dur
                    temp = spikes_cell[cond1&cond2]-self.off_triggers[i]
                    time_stamps_all[cell]['spikes_off'].append(temp)
                    off_count += len(temp[temp<off_threshold])
                    
                    cond1 = spikes_cell>self.on_triggers[i]
                    cond2 = spikes_cell<self.on_triggers[i]+self.stim_dur
                    temp = spikes_cell[cond1&cond2]-self.on_triggers[i]+self.stim_dur
                    time_stamps_all[cell]['spikes_on'].append(temp)
                    on_count += len(temp[temp < on_threshold])
            elif self.off_triggers[0]>self.on_triggers[0]:
#                 for i in range(len(self.off_triggers)):
#                     
#                     cond1 = spikes_cell>self.off_triggers[i]
#                     cond2 = spikes_cell<self.off_triggers[i]+self.stim_dur
#                     temp = spikes_cell[cond1&cond2]-self.off_triggers[i] 
#                     time_stamps_all[cell]['spikes_off'].append(temp)
#                     off_count += len(np.nonzero(cond1&cond2)[0])
#                     
#                     cond1 = spikes_cell>self.on_triggers[i]
#                     cond2 = spikes_cell<self.on_triggers[i]+self.stim_dur
#                     temp = spikes_cell[cond1&cond2]-self.on_triggers[i] + on_offset
#                     time_stamps_all[cell]['spikes_on'].append(temp)
#                     on_count += len(np.nonzero(cond1&cond2)[0])
                raise ValueError('Triggers gone wrong')
            else:
                raise ValueError('Triggers gone wrong')
               
            on_count = float(on_count)
            off_count = float(off_count)
            bi = (on_count-off_count)/(on_count+off_count)
            bias_index[cell] = bi
            thr_on = 0.5
            thr_off = -0.25
            if bi>thr_on:
                key = 'spikes_on'
                range_ = (500 ,1000)
                n_bins = 10
                temp = np.concatenate(time_stamps_all[cell][key])
                hist_, bins = np.histogram(temp, bins = n_bins, range = range_)
                hist_ = hist_ / np.max(hist_)
                ti = np.sum(hist_)/n_bins
            elif bi<thr_off:
                key = 'spikes_off'
                range_ = (0, 500)
                n_bins = 10
                temp = np.concatenate(time_stamps_all[cell][key])
                hist_, bins = np.histogram(temp, bins = n_bins, range = range_)
                hist_ = hist_ / np.max(hist_)
                ti = np.sum(hist_)/n_bins
                
            else:
                key = 'spikes_on'
                range_ = (500 ,1000)
                n_bins = 10
                temp = np.concatenate(time_stamps_all[cell][key])
                hist_, bins = np.histogram(temp, bins = n_bins, range = range_)
                hist_ = hist_ / np.max(hist_)
                ti = np.sum(hist_)/n_bins
                
                key = 'spikes_off'
                range_ = (0, 500)
                n_bins = 10
                temp = np.concatenate(time_stamps_all[cell][key])
                hist_, bins = np.histogram(temp, bins = n_bins, range = range_)
                hist_ = hist_ / np.max(hist_)
                ti = (ti+np.sum(hist_)/n_bins)/2
            transience_index[cell] = ti
                
            
                
            
        
        self.transience_index = transience_index
        self.bias_index = bias_index
        self.cell_to_loc = cell_to_loc
        self.structured_ts = time_stamps_all
    
    
    def raster(self, cell, ax = None, title= None, density = False, \
               cmap = 'Greys', sigma=15, axes_off=False, light_onset=True, **kwargs):
        """
        Creates a raster plot
    
        Inputs
        ----------
        cell    cell ID (used as key into self.structured_ts or self.robust_spikes)
        ax      matplotlib.pyplot axes object
                if passed, plot into these axes
        

        density    boolean    
                    if True, display a spike density estimate on top of the raster
        
        cmap        string
                    color map
        outputs
        -------
        none
        """
        
        if ax is not None:
            plt.sca(ax)
        spikes = self.structured_ts[cell]
        flattened_spikes = []
#         elif mode == 'tnt'
        for _, event_times_list in spikes.items():    
            for ith, trial in enumerate(event_times_list):
                flattened_spikes.append(trial)
                trial = trial/1000
                plt.vlines(trial, ith + .5, ith + 1.5, linewidths = 0.5, **kwargs)
                
            if light_onset:
                plt.vlines(0.5, .5, len(event_times_list)+.5, color='r', linewidth=1)
            plt.ylim(.5, len(event_times_list) + .5)
            plt.xlim(0, 1)
            if axes_off:
                plt.gca().axis('off')
            
        if density:
            spike_density =\
             np.concatenate(flattened_spikes)

            # determine the factor that is needed to arrive at a certain kernel
            # density sigma
            
            kdefactor = sigma/np.sqrt(np.cov(spike_density, rowvar=1, bias=False))
            g_kde = gaussian_kde(spike_density, bw_method = kdefactor)
#             print('The sd of the gaussian kernel for light spike density\
#                 estimation is {} ms'.format(np.sqrt(g_kde.covariance)))
            (x_min, x_max) = plt.gca().get_xlim()
            (y_min, y_max) = plt.gca().get_ylim()
            points = np.linspace(x_min, x_max*1000, x_max*1000)
            density_estimate = g_kde.evaluate(points)
            surf = np.tile(density_estimate, (len(event_times_list), 1))
            plt.imshow(surf, aspect = 'auto', cmap = cmap, \
                       extent=[x_min, x_max, y_min, y_max])

            if axes_off:
                plt.gca().axis('off')
        
        if title is not None:
            plt.title(title) 
        
                    

        