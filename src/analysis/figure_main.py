'''
Created on 14.11.2018

@author: larissa.hoefling@uni-tuebingen.de
'''


import h5py
import numpy as np
import matplotlib.pyplot as plt
# from analysis.utils import technical_figure
from analysis.stimulus import Stimulus
from sklearn.linear_model import LinearRegression
import pickle as pkl
from analysis.spikes import Spikes
from analysis.model import Model
from analysis.sta import SpikeTriggeredAnalysis
from analysis.stimulus import Stimulus
from analysis.utils import sigmoid, exponential
import matplotlib as mpl







def model_illustration_figure(cell, sta, stim,model, spikes, path = None,\
                              t_start = 0.63,t_stop = 0.67):
    
    
    '''
    ###########################################################################
    create ticklabel formatter
    '''
    def tick_formatter(value,tick_number):
        return (value - 20)/ 1000
    
    '''
    ##########################################################################
    create figure and axes
    '''
#     plt.style.use('seaborn-paper') 
    f = plt.figure() 
    plt.style.use('seaborn-paper') 
    s = 13
    mpl.rcParams.update({'font.size':s, 'axes.labelsize':s,\
                          'xtick.labelsize':s, 'ytick.labelsize':s})
    axes = []
    
    nrows = 2
    # the STA axis
    axes.append(plt.subplot2grid( [nrows,  2], [0, 1], rowspan=1,colspan=1,\
                                   fig=f))
    # the stim snippets axis
#     axes.append(plt.subplot2grid( [nrows, 4], [1, 0], rowspan=1, colspan=2, \
#                                   fig=f))
#     
    # the STE axis
    axes.append(plt.subplot2grid( [nrows, 2], [0, 0], rowspan=1, colspan=1, \
                                  fig=f))
    # the histograms axis
    axes.append(plt.subplot2grid([nrows, 2], [1, 0]))

    #the nonlinearity axis
    axes.append(plt.subplot2grid([nrows, 2], [1, 1], sharex = axes[2]))
    
    sta_xaxis = np.linspace(0, sta.len_sta_ms, sta.len_sta_samples)
    
    '''
    ###########################################################################
    get all necessary variables
    '''
    stim_ensemble = \
        stim.create_stimulus_ensemble(len_sta = sta.len_sta_samples, \
                            stepwidth = sta.len_sta_ms, white = sta.whitening)
    
    
    
    '''
    ###########################################################################
    plot all the stuff
    '''
    num_snippets = 100
    #plot the STA
    axes[0].plot(sta_xaxis, sta.sta[cell][0], color = 'k')
    axes[0].fill_between(sta_xaxis,sta.sta[cell][0]- sta.ste_stand_dev[cell][0],\
                         sta.sta[cell][0]+sta.ste_stand_dev[cell][0],\
                          color = 'k', alpha = 0.3)
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(tick_formatter))
    
    # plot the stim snippets and the STE snippets

    indices = np.round(spikes.structured_ts[cell][0]).astype(int) * 10**-3
    cond1 = indices>t_start
    cond2 = indices<t_stop
    indices = indices[np.logical_and(cond1, cond2)]
    
    time_line_start = indices[0]-sta.delta_t_to_spike_back*10**-3
    time_line_stop = indices[-1]-sta.delta_t_to_spike_forward*10**-3
    stim_sr = 10000
#     time_line = np.linspace(t_start-sta.delta_t_to_spike_back*10**-3,\
#                              t_stop-sta.delta_t_to_spike_forward*10**-3, \
#                              (t_stop-sta.delta_t_to_spike_forward*10**-3-\
#                               (t_start-sta.delta_t_to_spike_back*10**-3))*stim_sr)

    time_line = \
    np.linspace(time_line_start, 
                time_line_stop, 
                int(np.round(time_line_stop*stim_sr))-\
                int(np.round(time_line_start*stim_sr)))

#     axes[1].plot(time_line, \
#                  stim.white_current_density[int(t_start*stim_sr): int(t_stop*stim_sr)], \
#                  color = 'k', alpha = 0.5)
#     axes[1].plot(time_line, \
#                  stim.current_density[int(t_start*stim_sr): int(t_stop*stim_sr)], \
#                  color = 'k', alpha = 0.2)
        
   
#     axes[1].plot(time_line, \
#     stim.white_current_density[int(t_start*stim_sr-sta.delta_t_to_spike_back*10)\
#                                : int(t_stop*stim_sr-sta.delta_t_to_spike_forward*10)], \
#                  color = 'k', alpha = 0.5)
    axes[1].plot(time_line, \
                 stim.white_current_density[int(np.round(time_line_start*stim_sr)):\
                                             int(np.round(time_line_stop*stim_sr))],
                 color='k', alpha = 0.5)
    
    
#     axes[1].plot(time_line, \
#                  stim.current_density[int(t_start*stim_sr): int(t_stop*stim_sr)], \
#                  color = 'k', alpha = 0.2)
    axes[1].set_xlabel('Time [s]')
    
    axes[1].set_ylabel('Stimulus amplitude \n'r'[$\frac{mA}{cm^2}$]')
    ste_excerpts = []
    for idx in indices:
        
        current_idx = int(np.where(np.isclose(time_line, np.ones_like(time_line) * \
                              idx, atol = 0.0001))[0][0] + time_line_start*stim_sr)
        
        
        axes[1].vlines(idx, stim.white_current_density[current_idx]- 0.4, \
                       stim.white_current_density[current_idx]+ 0.4, color = 'k')
        x = np.linspace(idx - sta.delta_t_to_spike_back*10**-3, \
                        idx - sta.delta_t_to_spike_forward*10**-3, sta.len_sta_samples)
        
        stim_excerpt = stim.white_current_density[(current_idx -\
                             sta.delta_t_to_spike_back*10):(current_idx -\
                             sta.delta_t_to_spike_forward*10)]
        
        keyword_arguments = {'alpha' : 0.2}
#         axes[1].fill_between(x, stim_excerpt - 0.2, stim_excerpt +0.2, color = 'k',\
#                              **keyword_arguments)
        axes[0].plot(np.linspace(0, sta.len_sta_ms, len(stim_excerpt)), \
                     stim_excerpt, alpha = .2, color = 'k')
        ste_excerpts.append(stim_excerpt)
    axes[0].set_xlabel('Time [s]')
    (ymin, ymax) = axes[0].get_ylim()
    axes[0].vlines(20, ymin, ymax)
    # plot the histograms
    
    n_partition = 0
    edges = model.hist_bin_edges[cell][n_partition]
    hist_stim = model.hist_stim[cell][n_partition]
    hist_ste = model.hist_ste[cell][n_partition]
    width = edges[1] - edges[0]
    ax = axes[2]
    ax.bar(edges[:-1], hist_stim, width= width, align = 'edge', color = 'None',\
           edgecolor = 'k')
    ax.bar(edges[:-1], hist_ste/4, width= width,\
                align = 'edge', color = 'k', edgecolor = 'k')
    ax.axvline(x=0, color='k')
    ax.set_xlabel('Filter response')
    ax.set_ylabel('Number of \n stimuli')
#     ax.spines["bottom"].set_visible(False)

    # plot the nonlinearity
    
    
    ax = axes[3]
    ax.scatter(model.binned_linear_prediction[cell][n_partition], \
                            model.binned_nonlinear_prediction[cell][n_partition]/4, color = 'k')
    
    ax.axvline(x=0, color='k')
    ax.set_xlabel('Filter response')
    ax.set_ylabel('Histogram ratio')
#     ax.spines["bottom"].set_visible(False)

    
    r_sig = model.r_sig_all[cell][n_partition]
    r_exp = model.r_exp_all[cell][n_partition]
                
    if r_sig>r_exp:

        if model.success_sig_all[cell][n_partition] & (r_sig>0.7):
            sig_pred = sigmoid(model.binned_linear_prediction[cell][n_partition], *model.popt_sig_all[cell][n_partition])
            plt.plot(model.binned_linear_prediction[cell][n_partition], sig_pred, color = 'k', label = 'r_sig = '+str(r_sig))
#                 else:
#                     print('Sigmoid fit was not successfull for cell {}, partition {} (r_sig = {}) '.format(cell, n_partition, r_sig))
#                     
    else:
        if model.success_exp_all[cell][n_partition] & (r_exp>0.7):
            exp_pred = exponential(model.binned_linear_prediction[cell][n_partition], *model.popt_exp_all[cell][n_partition])
            
            plt.plot(model.binned_linear_prediction[cell][n_partition], exp_pred, color = 'r', label = 'r_exp = '+str(r_exp))
    
    
        
    for ax in axes:
        ax.spines["top"].set_visible(False)
#         ax.spines["bottom"].set_visible(False)
#         ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fname = 'modelIllustration'
    if path is not None:
        plt.savefig(path+fname+'.png', dpi = 300, fmt='png')
        plt.savefig(path+fname+'.pdf', dpi = 300, fmt='pdf')
        plt.savefig(path+fname+'.svg', dpi = 300, fmt='svg')
        plt.close()
        
    else:
        plt.show()


def technical_figure(raw_path, filtered_path, sorted_path, t_start, t_stop, \
                     x,y, stim, path = None, fname = 'ArtefactSubtraction',\
                     pre_spike = -0.02, post_spike =0.01, \
                     sta_illustration_axis = None):

    
    with h5py.File(raw_path, 'r') as raw_file:
         
        fs = 10**6/raw_file['Acquisition']['Sensor Data']['SensorMeta']['Tick'][0]
        exp =  raw_file['Acquisition']['Sensor Data']['SensorMeta']['Exponent'][0]
        cv = raw_file['Acquisition']['Sensor Data']['SensorMeta']\
        ['Conversion Factors'][0].decode('utf8').split(' ')
        cv = np.asarray(cv[:-1], dtype = np.float64).reshape(65,65)
        sensor_data = raw_file["Acquisition"]["Sensor Data"]["SensorData 1 1"]
        start = int(t_start*fs)
        stop = int(t_stop*fs)
        raw_trace = sensor_data[start:stop,x,y] * cv[x,y] * 10.0**exp * 10.0**3
        
#         
#         stim_trace = raw_file['Acquisition']['STG Waveform']['ChannelData 1'][0]\
#                     [int(start):int(stop)]
#         stg_cv = raw_file['Acquisition']['STG Waveform']['ChannelMeta']\
#                     ['ConversionFactor'][0]
#         stim_trace = stim_trace*stg_cv*10.0**exp *10.0**3
     
    with h5py.File(filtered_path, 'r') as filtered_file:
        
        sensor_data = filtered_file["Acquisition"]["Sensor Data"]\
        ["SensorData 1 1"]
        start = int(np.round((t_start-0.5),4)*fs)
        stop = int(np.round((t_stop-0.5),4)*fs)
        filtered_trace = \
        sensor_data[start:stop, x, y]* cv[x,y] * 10.0**exp * 10.0**3
        
    with h5py.File(sorted_path, 'r') as sorted_file:
        idx = np.where((sorted_file['Spike Sorter']['Units']['Column'] == x+1) & \
                       (sorted_file['Spike Sorter']['Units']['Row'] == y+1))
        unit = sorted_file['Spike Sorter']['Units']['UnitID'][idx][0]
        identifier = "Unit "+str(unit)
        source = sorted_file['Spike Sorter'][identifier]['Source'][start:stop]
        peaks = sorted_file['Spike Sorter'][identifier]['Peaks']
        mask = np.array(peaks['IncludePeak'], dtype=bool)
        ts = sorted_file['Spike Sorter'][identifier]['Peaks']['Timestamp'][mask]
        mask = np.array(peaks['IncludePeak'], dtype=bool)
#         spike_amplitudes = peaks['PeakAmplitude'][mask]
#         noise_amplitudes = peaks['PeakAmplitude'][np.invert(mask)]
#         threshold = (min(spike_amplitudes) + max(noise_amplitudes))/2
        threshold = peaks.attrs['AmplitudeThreshold'][0]
    time_line = np.linspace(t_start, t_stop, np.round(t_stop-t_start,4)*fs)

    cond1 = ts>(t_start-0.5) * 10**6
    cond2 = ts<(t_stop-0.5) * 10**6
    
    spikes = ts[np.logical_and(cond1, cond2)]
    spikes = spikes*10**-6 + 0.5
    idx = [np.where(np.isclose(time_line, np.ones_like(time_line) * \
                              spike, atol = 0.0001))[0][0] for spike in spikes]
    
#     corr = correlate(stim.voltage_command[int(start/2):int(stop/2)], 
#                      raw_trace[::2], mode = 'same')
#     expected_peak = np.int(len(raw_trace[::2])/2)
#     max_lag = expected_peak-np.argmax(corr)

    x1 = stim.voltage_command[int(start/2)%50000:int(stop/2)%50000]
    x2 = stim.current_density[int(start/2)%50000:int(stop/2)%50000]
    predictors = np.transpose([x1,x2])
    to_predict = raw_trace[::2]
    
    
    '''
    ######################################################################
    start plotting
    '''
    f = plt.figure()
#     plt.style.use('seaborn-poster') 
    s = 13
    mpl.rcParams.update({'font.size':s, 'axes.labelsize':s,\
                          'xtick.labelsize':s, 'ytick.labelsize':s})
#     f = plt.figure(figsize = [20,16])
    

    axes = []
#     f, axes = plt.subplots(nrows=5, ncols=1, figsize = [16,12])

    nrows = 3
    ncols = 4
    axes.append(plt.subplot2grid(shape=[nrows, ncols], loc =[0, 0], rowspan=1, \
                                 colspan=4, fig=f))
    axes.append(plt.subplot2grid(shape=[nrows, ncols], loc =[1, 0], rowspan=1,\
                                  colspan=4, fig=f, sharex = axes[0]))
    axes.append(plt.subplot2grid(shape=[nrows, ncols], loc =[2, 0], rowspan=1, \
                                 colspan=4, fig=f, sharex = axes[0]))
#     axes.append(plt.subplot2grid(shape=[nrows, ncols], loc =[3, 0], rowspan=1, \
#                                  colspan=4, fig=f, sharex = axes[0]))

    current_excerpt = stim.current_density[int(start/2):int(stop/2)]
    white_current_excerpt = stim.white_current_density[int(start/2):int(stop/2)]
#     l1 = ax.plot(time_line[::2], white_current_excerpt, \
#                  label = 'whitened stimulus', color = 'k', linestyle = '-.')
    ax = axes[0]
    l2 = ax.plot(time_line[::2], current_excerpt, \
            label = 'current density', color = 'k', linewidth = 2)

    
    lines = l2
    labels = [l.get_label() for l in lines]

    ax.set_ylabel('Current density \n' r'[$\frac{mA}{cm^2}$]')
    
    ax.set_ylim([-2,2])
#     ax.yaxis.tick_right()
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

#     axes[0].set_ylabel('Voltage [mV]')
#     axes[0].set_title('Current density')
    
    '#######################################################################'
    axes[1].plot(time_line,raw_trace, color = 'k')
#     axes[1].set_ylabel('Voltage [mV]')
#     axes[1].set_title('Raw signal')
#     axes[1].yaxis.tick_right()

#     axes[1].set_ylim([np.min(raw_trace), 0])
#     axes[1].set_ylim([-10,1])
    axes[1].set_ylabel('Voltage \n [mV]')
    '#######################################################################'
    axes[2].plot(time_line,filtered_trace, color = 'k')
    
#     axes[2].set_ylabel('Voltage [mV]')
#     axes[2].set_title('Artefact subtracted and filtered')
#     plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Voltage \n [mV]')
    '#######################################################################'
#     axes[3].plot(time_line, source, color = 'k')
#     axes[3].plot(time_line, np.ones_like(time_line) * threshold)
#     axes[3].set_ylabel('Source signal [AU]')
# #     axes[3].set_title('Spike sorter source signal')
#     axes[3].set_xlabel('Time [s]')
#     plt.gca().yaxis.tick_right()
    for ax in axes[:-1]:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
#         ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.labelpad = 10
        for label in ax.get_xticklabels():
            label.set_visible(False)
        
    axes[-1].spines["top"].set_visible(False)
#     axes[-1].spines["left"].set_visible(False)
    axes[-1].spines["right"].set_visible(False)
#     plt.text(0.03, 0.45, 'Voltage [mV]', transform = f.transFigure, \
#              rotation = 'vertical')
#     plt.tight_layout()
#     mpl.rcParams.update({'font.size':s, 'axes.labelsize':s,\
#                           'xtick.labelsize':s, 'ytick.labelsize':s})
    f.subplots_adjust(left = 0.2)
    if path is not None:
#         plt.savefig(path+fname+'.ps', dpi = 300, fmt='ps')
        plt.savefig(path+fname+'.png', dpi = 300, fmt='png')
        plt.savefig(path+fname+'.pdf', dpi = 300, fmt='pdf')
        plt.savefig(path+fname+'.svg', dpi = 300, fmt='svg')
        plt.close()
#     plt.show()
    
    
    
    
plt.style.use('seaborn-paper')
# mpl.rcParams.update({'font.size':13, 'axes.labelsize':13,\
#                           'xtick.labelsize':13, 'ytick.labelsize':13})
# raw_path = r"D:/Lara Analysis/2018-10-30 - bl6/raw/2018.10.30-14.26.56-standard_hf.cmcr"
#  
# filtered_path = r'D:/Lara Analysis/2018-10-30 - bl6/raw/2018.10.30-14.26.56-standard_hf_mod_f.cmcr'
# sorted_path =  r'D:/Lara Analysis/2018-10-30 - bl6/sorted/2018.10.30-14.26.56-standard_hf_mod_f_appended_GZ_checked.cmtr'

figure_path = r'/gpfs01/euler/User/lhoefling/estim_paper/paper figures/'
raw_path = r'/gpfs01/euler/User/lhoefling/estim_paper/2017-11-03 - bl6/2017.11.03-13.10.36-standard_subfield_nofilter.cmcr'
   
filtered_path = r'/gpfs01/euler/User/lhoefling/estim_paper/2017-11-03 - bl6/2017.11.03-13.10.36-standard_subfield_nofilter_mod_f.cmcr'
sorted_path = r'/gpfs01/euler/User/lhoefling/estim_paper/2017-11-03 - bl6/2017.11.03-13.10.36-standard_subfield_nofilter_mod_f_appended_1911.cmtr'
# 
#stim_path = r'D:\Lara analysis\stim\WGNdur5_fs10000_reflect10_cutoff_100_2017_04_12_10.17.09.txt'
stim_path = r'/gpfs01/euler/User/lhoefling/estim_paper/stim/WGNdur5_fs10000_reflect10_cutoff_100_2017_04_12_10.17.09.txt'
my_stim = Stimulus(stim_path)
with open(r'/gpfs01/euler/User/lhoefling/estim_paper/stim/pickled_stim_obj.pkl', 'wb') as f:
    pkl.dump(my_stim, f)
technical_figure(raw_path,filtered_path, sorted_path, t_start = 1.1, t_stop = 1.25, \
                 x = 23-1, y = 14-1, stim = my_stim, path = figure_path, fname = '3')
 

# filtered_path = r"Z:/Lara/Measurements_10_2017/2017-12-19 - bl6/2017.12.19-15.05.43-standard_sf_mod_lara_appended.cmcr"
# raw_path = r"Z:/Lara/Measurements_10_2017/2017-12-19 - bl6/2017.12.19-15.05.43-standard_sf.cmcr"
# sorted_path = r"D:/Lara Analysis/2017-12-19 - bl6/sorted/2017.12.19-15.05.43-standard_sf_mod_lara_appended_high_sensitivity_checked.cmtr"


# raw_path = r'D:\Lara analysis\2017-12-07 - rd10\raw\2017.12.07-17.25.47-standard.cmcr'
# filtered_path = r'D:\Lara analysis\2017-12-07 - rd10\raw\2017.12.07-17.25.47-standard_mod_nf_appended.cmcr'
# sorted_path = r'D:\Lara analysis\2017-12-07 - rd10\sorted\2017.12.07-17.25.47-standard_mod_nf_appended_c.cmtr'

# 
# sorted_path = r'D:\Lara analysis\2017-10-26 - rd10\sorted\2017.10.26-17.13.58-standard_subfield_mod_f_appended.cmtr'
# raw_path = r'D:\Lara analysis\2017-10-26 - rd10\raw\2017.10.26-17.13.58-standard_subfield.cmcr'
# filtered_path =  r'D:\Lara analysis\2017-10-26 - rd10\raw\2017.10.26-17.13.58-standard_subfield_mod_f.cmcr'

# sorted_path =  r'D:\Lara analysis\2017-11-09 - rd10\sorted\2017.11.09-14.24.52-standard_subfield_mod_f_appended_c.cmtr'
# raw_path = r'D:\Lara analysis\2017-11-09 - rd10\raw\2017.11.09-14.24.52-standard_subfield.cmcr'
# filtered_path = r'D:\Lara analysis\2017-11-09 - rd10\raw\2017.11.09-14.24.52-standard_subfield_mod_f.cmcr'


# sorted_path = r'D:\Lara analysis\2018-04-26 - rd10\sorted\2018.04.26-13.39.51-standard_sf_mod_f_appended_c.cmtr'
# raw_path = r'D:\Lara analysis\2018-04-26 - rd10\raw\2018.04.26-13.39.51-standard_sf.cmcr'
# filtered_path = r'D:\Lara analysis\2018-04-26 - rd10\raw\2018.04.26-13.39.51-standard_sf_mod_f.cmcr'


sorted_path = r'D:\Lara analysis\2018-07-19 - rd10\sorted\2018.07.19-15.02.43-standard_sf_mod_lara_appended.cmtr'
raw_path = r'D:\Lara analysis\2018-07-19 - rd10\raw\2018.07.19-15.02.43-standard_sf.cmcr'
filtered_path =  r'D:\Lara analysis\2018-07-19 - rd10\raw\2018.07.19-15.02.43-standard_sf_mod_lara.cmcr'
with h5py.File(sorted_path) as f:
    units = f['Spike Sorter']['Units']['UnitID']
    cols = f['Spike Sorter']['Units']['Column']
    rows = f['Spike Sorter']['Units']['Row']
    cell_to_loc = {unit: (cols[i], rows[i]) for i, unit in enumerate(units)}
# for unit, loc in cell_to_loc.items():
#     technical_figure(raw_path,filtered_path, sorted_path, t_start = 1, t_stop = 1.25, \
#                  x = loc[0]-1, y = loc[1]-1, stim = my_stim, path = figure_path,\
#                  fname = '2018.07.19-15.02.43_'+str(unit))
    
     
f = open(r'D:/Lara Analysis/bl6 joined/pickled/joined_white_STA.pkl', 'rb')
my_sta = pkl.load(f)
 
 

 
f = open(r'D:/Lara Analysis/bl6 joined/pickled/model.pkl', 'rb')
my_model = pkl.load(f)
 
f = open(r'D:/Lara Analysis/bl6 joined/pickled/joined_spikes.pkl', 'rb')
my_spikes = pkl.load(f)

model_illustration_figure(cell=103, sta = my_sta, stim = my_stim, model = my_model\
                          , spikes = my_spikes, path = figure_path,t_start = 0.80,t_stop = 0.855)

my_stim.check_and_plot(len_snippets=300, stepwidth_snippets=30, \
                        path = figure_path, 
                       load_pickled_version =True)





if __name__ == '__main__':
    pass