'''
Created on 13.08.2018

@author: larissa.hoefling@uni-tuebingen.de
-*- coding: utf-8 -*-
'''
import numpy as np
import matplotlib.pyplot as plt
from analysis.utils import smooth
from scipy.signal import periodogram
import pickle as pkl
import matplotlib as mpl


class Stimulus():
    '''
    #########################################################################
    When an object of the class stimulus is instantiated, a stimulus file (.txt)
    is opened and the timeline and voltage commands are read, and the expected 
    current is computed. For now, a general capacitance value is assumed for all
    chips of the same type (typically no oxide), however in a future version, a 
    chip name should be passed and the corresponding exact capacitance should be
    retrieved from a .csv file and saved as an instance variable 
    '''
    temp_res = 10000
    conversion_factor = 10**(-3) #times 10**-6 to convert from uF to F, 
                                 #times 10**-6 to convert from uV to V,
                                 #times 10**4 to convert from 1/0.0001 s to 1/s
                                 #times 10**3 to convert from A to mA
                                 #times 10**2 to convert from 1/mm**2 to 1/cm**2
    

    def __init__(self, path = None, len_snippets = 200, stepwidth_snippets=20, \
                 rcond=10**-5, smoothing_window_len = 20):
        '''
        set the path to the stimulus file, set the capacitance that should be 
        used to convert from voltage command to expected current density;
        
        load the stimulus time_line and the voltage_command from the .txt file
        found under path
        
        compute the expected current density 
        
        Inputs:
        -----------------------------------------------------------------------
        path    path to stimulus file
        len_snippets    int       
        stepwidth_snippets    int
        rcond                 float
        smoothing_window_len    int
                parameters to reconstruct_stimulus
                
        Outputs:
        -----------------------------------------------------------------------
        None;
        sets
        self.path to path
        self.capacitance to fixed value
        self.time_line to the time line read from file
        self.voltage_command to the voltage command read from file
        self.current_density to the output of self.expected_current()
        self.white_current_density to the output of self.reconstruct_stimulus()
        '''
        if path == None:
            #fill this later with a gui
            raise ValueError("no path for the stimulus file was passed")
        else:
            self.path = path
         
        self.capacitance = 0.04227
        temp = np.genfromtxt(path)
        
        self.time_line = temp[:-1,0]
        self.voltage_command = temp[:,1]
        self.current_density = self.expected_current()
        self.white_current_density = \
            self.reconstruct_stimulus(len_snippets, stepwidth_snippets, rcond, smoothing_window_len)
#         self.white_current_density = \
#         np.concatenate(np.transpose(self.whiten_stimulus(len_snippets, stepwidth_snippets, rcond)))
        

        
    def expected_current(self):
        '''
        using I = C*dV/dt and i = I/A, compute the expected current density 
        '''
        current_density = \
        np.diff(self.voltage_command)*self.capacitance*self.conversion_factor
        return current_density
    
    
    def create_stimulus_ensemble(self, len_sta, stepwidth = 1, white = False):
        '''
        from the continuous stimulus loaded from the file, create a stimulus 
        ensemble, where each element is a len_sta long snippet, and snippets are
        taken stepwidth ms apart 
        
        input: 
        len_sta     int
                    length of the snippets in samples
        stepwidth    int
                    stepwidth between the snippets in ms 
        white        boolean
            if white is true, then the stimulus ensemble is created from the 
            reconstructed whitened stimulus
        
        output:
        stim_ensemble    array
                        the created stimulus ensemble, dimensions 
                        len_sta x # of snippets (i.e. 1 column is 1 snippet)
        
        
        '''
        stim_ensemble = []
        sample_stepwidth = int((self.temp_res/1000)*stepwidth)
        if white:
            
            len_stim = len(self.white_current_density)
            dim = len_stim-len_sta
            [stim_ensemble.append(self.white_current_density[i:i+len_sta]) \
                                    for i in range(0, dim+sample_stepwidth, sample_stepwidth)]
        else:
            
            len_stim = len(self.current_density)
            dim = len_stim-len_sta
            [stim_ensemble.append(self.current_density[i:i+len_sta]) \
                                    for i in range(0, dim+sample_stepwidth, sample_stepwidth)]

            
        # if len(stim_ensemble[-1]) != len(stim_ensemble[-2]):
        #     stim_ensemble.pop(-1)
        #     print("Last element of stimulus ensemble had to be removed because it was of a different dimension ")
        if stepwidth==1:
            stim_ensemble.pop(-1)
        stim_ensemble = np.asarray(stim_ensemble)
        stim_ensemble = np.transpose(stim_ensemble)
        return stim_ensemble
    
    def compute_stimulus_cov(self, len_sta, mode = 'original'):
        '''
        computes the surrogate stimulus covariance, based on the stimulus 
        ensemble with snippets of length len_sta (for more details, see 
        description of create_stimulus_ensemble)
        
        '''
        if mode == 'original':
            stim_ensemble = self.create_stimulus_ensemble(len_sta)
        elif mode == 'white_smooth':
            stim_ensemble = self.create_stimulus_ensemble(len_sta, white=True)
        elif mode == 'white_raw':
            stim_ensemble = self.whiten_stimulus(len_sta, stepwidth_stim_snippets=1)
            
        stim_cov = np.cov(stim_ensemble)
     
        return stim_cov
    
    def compute_stim_cov_inverse_sqrt(self, len_sta, rcond=10**-5):
        '''
        computes sqrt pseudoinverse of the stimulus covariance matrix. For 
        details, see Aljadeff et al. (2016), Neuron
        '''
        stim_cov = self.compute_stimulus_cov(len_sta)
        u,s,v = np.linalg.svd(stim_cov)
        singular_values = s
        rcond = rcond * np.max(singular_values)
        ind = np.where(singular_values<=rcond)
        singular_values = 1/np.sqrt(singular_values)
        singular_values[ind] = 0
        sqrt_inv = np.dot(np.dot(u, np.diag(singular_values)), v)
        return sqrt_inv
    
    def compute_stim_cov_inverse(self, len_sta, rcond=10**-5):
        '''
        computes pseudoinverse of the stimulus covariance matrix. For 
        details, see Aljadeff et al. (2016), Neuron
        '''
        stim_cov = self.compute_stimulus_cov(len_sta)
        u,s,v = np.linalg.svd(stim_cov)
        singular_values = s
        rcond = rcond * np.max(singular_values)
        ind = np.where(singular_values<=rcond)
        singular_values = 1/singular_values
        singular_values[ind] = 0
        inv = np.dot(np.dot(u, np.diag(singular_values)), v)
        return inv
    
    def whiten_stimulus(self, len_sta, stepwidth_stim_snippets, rcond = 10**-5):
        '''
        creates a stimulus ensemble, and whitens it, i.e. removes autocorrelations
        from this ensemble by multiplying with the sqrt of the pseudo inverse
        of the stimulus covariance matrix
        '''
        
        stim_cov_sqrt_inv = self.compute_stim_cov_inverse_sqrt(len_sta, rcond)
        stimulus_ensemble = self.create_stimulus_ensemble(len_sta, stepwidth_stim_snippets)
        whitened_stimulus_ensemble = \
        np.matmul(stim_cov_sqrt_inv, stimulus_ensemble)
        
        
        return whitened_stimulus_ensemble
    
    

    def reconstruct_stimulus(self, len_snippets, stepwidth_snippets, rcond, \
                             smoothing_window_len):
        '''
        This function expects a stimulus ensemble of shape snippet_length x 
        snippet_number, where the snippets just have to be appended to yield
        the full stimulus (i.e. snippet_length * snippet_number = stimulus
        length), but the transitions need to be smoothed. The smoothing is 
        performed here. 
        
        Inputs:
        -----------------------------------------------------------------------
            len_snippets    int
                            length of the stimulus snippets in samples
        stepwidth_snippets    int
                                stepwidth between the snippets in ms; 
        rcond            float
                         rounding condition to compute_stim_cov_inverse_sqrt
        smoothing_window_len    int
                                length of the smoothing window to smooth, in samples
                                
        Outputs:
        -----------------------------------------------------------------------
        the whitened stimulus, reconstructed by appending and smoothing the 
        snippets from the whitened ensemble 
        '''
        
        
#         stim_ensemble = []
#         sample_stepwidth = int((self.temp_res/1000)*stepwidth_snippets)
#         
#             
#         dim = len(self.current_density)
# 
#         [stim_ensemble.append(self.current_density[i:i+len_snippets]) \
#                                         for i in range(0, dim, sample_stepwidth)]
# 
#             
#         
#         stim_ensemble = np.asarray(stim_ensemble)
#         stim_ensemble = np.transpose(stim_ensemble)


        stim_ensemble = self.create_stimulus_ensemble(len_snippets, stepwidth_snippets)
        stim_cov_inv_sqrt = self.compute_stim_cov_inverse_sqrt(len_snippets, rcond)
        white_ensemble = np.matmul(stim_cov_inv_sqrt, stim_ensemble)
        shift = int(smoothing_window_len/2)
        recon = np.concatenate(np.transpose(white_ensemble))

        smooth_recon = smooth(recon, window_len=smoothing_window_len)
        return smooth_recon
        
    def check_and_plot(self, len_snippets = 200, stepwidth_snippets = 20,\
                        rcond = 10**-5, path = None, load_pickled_version = False):
        
        

        '''
        creates a 4x3 plot of 
                    (1) an excerpt from x
                    (2) the periodogram of x
                    (3) the covariance of x
                    (4) the autocorrelation of x
        where x
                    (1) the original stimulus
                    (2) the whitened and smoothed version of the stimulus 
                    (3) the whitened raw version of the stimulus
        '''
        
        
        def format_func(value, tick_number):
            return value/self.temp_res
        
        
        
        
        
        fname = 'stimulus'
        if load_pickled_version:
            f = open(r'D:/Lara Analysis/paper pickles/stim_versions.pkl', 'rb')
            stim_versions = pkl.load(f)
            f = open(r'D:/Lara Analysis/paper pickles/cov_versions.pkl', 'rb')
            cov_versions = pkl.load(f)
            
        else:
            original = self.current_density
            original_cov = self.compute_stimulus_cov(len_snippets, mode = 'original')
            whitened_smoothed = self.white_current_density
            whitened_smoothed_cov = self.compute_stimulus_cov(len_snippets, mode='white_smooth')
            whitened_raw = np.concatenate(np.transpose(self.whiten_stimulus(len_snippets, stepwidth_snippets, rcond)))
            whitened_raw_cov = self.compute_stimulus_cov(len_snippets, mode='white_raw')
    #         stim_versions = dict(zip(['original', 'whitened and smoothed', 'whitened raw'], \
    #                                  [original, whitened_smoothed, whitened_raw]))
    #         
    #         cov_versions = dict(zip(['original', 'whitened and smoothed', 'whitened raw'], \
    #                                  [original_cov, whitened_smoothed_cov, whitened_raw_cov]))
            
            
            stim_versions = dict(zip(['original', 'whitened and smoothed'], \
                                     [original, whitened_smoothed]))
            
            cov_versions = dict(zip(['original', 'whitened and smoothed'], \
                                     [original_cov, whitened_smoothed_cov]))
        
            f = open(r'D:/Lara Analysis/paper pickles/stim_versions.pkl', 'wb')
            pkl.dump(stim_versions, f)
            f = open(r'D:/Lara Analysis/paper pickles/cov_versions.pkl', 'wb')
            pkl.dump(cov_versions, f)
        
        
        
        fs = self.temp_res
        n_cols = 1
        n_rows = 3
        counter = 0
        
        sample_start = 1000
        sample_stop = 2500
        t_start = sample_start/fs
        t_stop = sample_stop/fs
        
        time_line = np.linspace(t_start, t_stop, sample_stop-sample_start)
        
        
        '''
        plot original stimulus
        '''
        
#         plt.figure(figsize=[6,8])
#         plt.figure(figsize=[6.4,8])
        plt.figure(figsize=[8,6])
        plt.style.use('seaborn-paper')
        s=13
        mpl.rcParams.update({'font.size':s, 'axes.labelsize':s,\
                          'xtick.labelsize':s, 'ytick.labelsize':s})
        axes_orig = []
        
#         axes_orig.append(plt.subplot2grid([n_rows, 3], [1, 0]))
#         ax = plt.imshow(cov_versions['original'], cmap = 'Greys')
#         plt.gca().set_axis_off()
# #         plt.colorbar(ticks = [-0.1, 0, 0.3])
#         
# 
#         
#         axes_orig.append(plt.subplot2grid([n_rows, 3], [1, 1]))
#         plt.imshow(self.compute_stim_cov_inverse_sqrt(len_sta=len_snippets, \
#                                                  rcond = rcond), cmap = 'Greys')
#         plt.gca().set_axis_off()
# #         plt.colorbar(ticks = [-2, 0, 5])
#         
#         axes_orig.append(plt.subplot2grid([n_rows, 3], [1, 2]))
#         plt.imshow(cov_versions['whitened and smoothed'], cmap = 'Greys')
#         plt.gca().set_axis_off()
#         plt.colorbar(ticks = [0, 0.02, 0.05])
        
        axes_orig.append(plt.subplot2grid([n_rows, n_cols], [0, counter]))
        plt.plot(time_line, stim_versions['original'][sample_start:sample_stop],\
                 color = '0.5')

        
        plt.plot(time_line, stim_versions['whitened and smoothed'][sample_start:sample_stop],\
                 color = 'k')
        plt.xlabel('Time [s]')
        plt.ylabel('Current \n'  r'density [$\frac{mA}{cm^2}$]')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
        plt.gca().yaxis.tick_right()
        
        ax = plt.subplot2grid([n_rows, n_cols], [1, counter])
        
        start = 20
        
        axes_orig.append(ax)
        '''
        compute the absolute current in Amperes
        '''
        area = 3*((5*0.003)**2)                #active stimulation area, assuming 3 x 5x5 active electrodes, one
                                              #electrode has 0.003 cm length

        orig_stim = stim_versions['original'] * area * 10**-3   #last factor to convert from mA to A
        whitened_smoothed_stim = stim_versions['whitened and smoothed'] * area * 10**-3

        frequencies, psd = periodogram(orig_stim, fs, return_onesided = True)
#         bax = brokenaxes(ylims = ((psd[0], psd[22]),(psd[23], psd[2000])), subplot_spec=psd_ax)
        plt.semilogy(frequencies[start:2000], psd[start:2000], color = '0.5')
#         plt.yscale(value='log')
        frequencies, psd = periodogram(whitened_smoothed_stim, fs, return_onesided = True)
#         bax = brokenaxes(ylims = ((psd[0], psd[20]),(psd[21], psd[2000])), subplot_spec=psd_ax)
        plt.semilogy(frequencies[start:2000], psd[start:2000], color = 'k')
#         plt.yscale(value='log')
#         plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlim((frequencies[start], frequencies[2000]))
        current_ticks = list(plt.xticks()[0])
        current_ticks.append(4)
        current_ticks.pop(0)
        plt.xticks(current_ticks)
        plt.gca().yaxis.tick_right()
        print(frequencies[20])
#         plt.gca().yaxis.tick_right()
#         plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.xlabel('Frequency [Hz]')
#         plt.gca().yaxis.set_label_position('right')
        plt.ylabel('Power spectral' +'\n' +r'density [$\frac{A}{\sqrt{Hz}}$]', labelpad = 14)
        
        axes_orig[1].spines["top"].set_visible(False)
#         axes_orig[1].spines["bottom"].set_visible(False)
        axes_orig[1].spines["left"].set_visible(False)
        axes_orig[1].spines["right"].set_visible(False)
#         plt.semilogy(frequencies[20:2000], psd[20:2000])
#         axes_orig.append(plt.subplot2grid([n_rows, n_cols+1], [2, counter]))
#         
#         plt.imshow(cov_versions['original'],cmap = 'hot')
#         
#         axes_orig.append(plt.subplot2grid([n_rows, n_cols+1], [2, 1]))
#         
#         plt.imshow(cov_versions['whitened and smoothed'], cmap='hot')
#         
#         axes_orig[2].spines["top"].set_visible(False)
#         axes_orig[2].spines["bottom"].set_visible(False)
#         axes_orig[2].spines["left"].set_visible(False)
#         axes_orig[2].spines["right"].set_visible(False)
#         axes_orig[2].xaxis.set_major_locator(plt.NullLocator())
#         axes_orig[2].yaxis.set_major_locator(plt.NullLocator())
#         
#         axes_orig[3].spines["top"].set_visible(False)
#         axes_orig[3].spines["bottom"].set_visible(False)
#         axes_orig[3].spines["left"].set_visible(False)
#         axes_orig[3].spines["right"].set_visible(False)
#         axes_orig[3].xaxis.set_major_locator(plt.NullLocator())
#         axes_orig[3].yaxis.set_major_locator(plt.NullLocator())
        
        axes_orig.append(plt.subplot2grid([n_rows, n_cols], [2, counter]))
        plt.acorr(stim_versions['original'], maxlags = 200, color = '0.5', \
                  usevlines= False, linewidth = 1, linestyle = '-')
        
        plt.acorr(stim_versions['whitened and smoothed'], maxlags = 200, \
                  color = 'k', usevlines = False, linestyle = '-')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Auto-\n correlation')
        axes_orig[-1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
        plt.gca().yaxis.tick_right()
        for ax in axes_orig:
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
        '''
        plot whitened and smoothed stimulus
        '''
#         counter = 1
#         axes_ws = []
#         axes_ws.append(plt.subplot2grid([n_rows, n_cols], [0, counter], sharey= axes_orig[0]))
#         plt.plot(time_line, stim_versions['whitened and smoothed'][sample_start:sample_stop])
#         
#         axes_ws.append(plt.subplot2grid([n_rows, n_cols], [1, counter]))
#         frequencies, psd = periodogram(stim_versions['whitened and smoothed'], fs, return_onesided = True)
#         plt.plot(frequencies[:2000], psd[:2000])
#         axes_ws[1].spines["top"].set_visible(False)
#         axes_ws[1].spines["bottom"].set_visible(False)
#         axes_ws[1].spines["left"].set_visible(False)
#         axes_ws[1].spines["right"].set_visible(False)
# #         plt.semilogy(frequencies[20:2000], psd[20:2000])
#         axes_ws.append(plt.subplot2grid([n_rows, n_cols], [2, counter]))
#         
#         plt.imshow(cov_versions['whitened and smoothed'])
#         axes_ws[2].spines["top"].set_visible(False)
#         axes_ws[2].spines["bottom"].set_visible(False)
#         axes_ws[2].spines["left"].set_visible(False)
#         axes_ws[2].spines["right"].set_visible(False)
#         axes_ws[2].xaxis.set_major_locator(plt.NullLocator())
#         axes_ws[2].yaxis.set_major_locator(plt.NullLocator())
#         
#         axes_ws.append(plt.subplot2grid([n_rows, n_cols], [3, counter]))
#         plt.acorr(stim_versions['whitened and smoothed'], maxlags = 200)
#         axes_ws[-1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        
        
        
        
        
        
        
        
#         plt.tight_layout()  
        
        plt.subplots_adjust(bottom = 0.2,wspace = 0.3, hspace = 1)
        if path is not None:
            plt.savefig(path+fname+'.png', dpi = 300, fmt='png')
            plt.savefig(path+fname+'.pdf', dpi = 300, fmt='pdf')
            plt.savefig(path + fname + '.svg', dpi=300, fmt='svg')
        else:
            plt.show()
        
        
        
        
        
        
    