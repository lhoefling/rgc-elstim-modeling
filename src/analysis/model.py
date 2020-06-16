'''
Created on 27.07.2018

@author: larissa.hoefling@uni-tuebingen.de
'''
import numpy as np
from scipy.optimize import curve_fit, nonlin
from analysis.utils import sigmoid, exponential, test_goodness_of_fit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from analysis.stimulus import Stimulus
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

class Model():
    '''
    This class performs the necessary steps to model cell responses to electrical
    stimuli.
    ###########################################################################
    attributes:
    
    set in create_stimulus_ensemble:
        used_stimulus_ensemble    saves the stimulus ensemble that was used to 
                                create the linear prediction
        cells                    saves the cells for which modeling was performed
        linear_prediction        saves a dictionary with cells as keys and a list 
                                of linear predictions for all stimulus snippets, 1 for each partition
        binned_linear_prediction    saves a dictionary with cells as keys and a list 
                                of linear predictions for the binned stimulus snippets, 1 for each partition 
        nonlinear_prediction    corresponding nonlinear_predictions
    
    set in fit_nonlinear_function
        popt_sig_all
        r_sig_all
        popt_exp_all
        r_exp_all
        success_sig_all
        success_exp_all
        for details see fit_nonlinear_function()
    ###########################################################################
    methods
    create_linear_prediction(self, spikeTriggeredAnalysis,  cells = None)
    fit_nonlinear_function
    fit_sigmoid
    fit_exponential
    plot_nonlinearity
    
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.initialized = True
        
     
    def create_prediction(self, spikeTriggeredAnalysis,  cells = None):
        
        '''
        This method creates the linear prediction for the given cells, based on
        the partitioned sta found in spikeTriggeredAnalysis.sta. To this end,
        it first gets the stimulus ensemble that was used to compute the STA
        . It then computes the dot product between the partition
        STAs and the stimulus ensemble (lin_pred_all). 
        To create an estimate of the nonlinearity
        function for each cell and each partition, the linear prediction for the
        spike-triggered ensemble (i.e. a subset of the stimulus ensemble,
        taken from spikeTriggeredAnalysis.ste) is 
        computed as well (lin_pred_st). lin_pred_st and lin_pred_all are binned
        using the same bin edges, and the ratio between lin_pred_st and 
        lin_pred_all is computed to estimate the shape of the nonlinearity 
        (see Schwartz et al. (2006)).
        
        
        Inputs:
        -----------------------------------------------------------------------
        spikeTriggeredAnalysis    object of class SpikeTriggeredAnalysis, 
                                  initialized
        
        Outputs:
        -----------------------------------------------------------------------
        None, sets:
        self.used_stimulus_ensemble   
        self.cells    list of cells 
        self.linear_prediction    dictionary
                                  with cell IDs as keys and a list of length
                                  n_partition, where the nth entry contains the 
                                  linear prediction of the model for each 
                                  stimulus in the nth training partition
        self.binned_linear_prediction    dictionary
                                         with cell IDs as keys and a list of 
                                         length n_partition, where the nth entry
                                         contains the binned version of the nth
                                         entry in self.linear_prediction
        self.binned_nonlinear_prediction    dictionary
                                            with cell IDs as keys and a list 
                                            of length n_partition, where the nth
                                            entry contains the binned nonlinear
                                            prediction for the stimuli in the
                                            nth training partition, that is 
                                            the ratio between the hist_st and 
                                            hist_all
        '''
        
        
        assert hasattr(spikeTriggeredAnalysis, 'sta'), "spikeTriggeredAnalysis \
        object has not been initialized"

        stimulus_snippets = spikeTriggeredAnalysis.used_stimulus_ensemble
            
        self.used_stimulus_ensemble = stimulus_snippets

        if cells == None: 
            cells = list(spikeTriggeredAnalysis.sta.keys())
        self.cells = cells
            
        linear_prediction = \
        {cell:[[] for _ in range(len(spikeTriggeredAnalysis.sta[cell]))] \
                                   for cell in cells}
        binned_linear_prediction = \
        {cell:[[] for _ in range(len(spikeTriggeredAnalysis.sta[cell]))] \
                                   for cell in cells}
        binned_nonlinear_prediction = \
        {cell:[[] for _ in range(len(spikeTriggeredAnalysis.sta[cell]))] \
                                   for cell in cells}
        hist_ste_all = \
        {cell:[[] for _ in range(len(spikeTriggeredAnalysis.sta[cell]))] \
                                   for cell in cells}
        hist_stim_all = \
        {cell:[[] for _ in range(len(spikeTriggeredAnalysis.sta[cell]))] \
                                   for cell in cells}
        
        
        hist_bin_edges_all = \
        {cell:[[] for _ in range(len(spikeTriggeredAnalysis.sta[cell]))] \
                                   for cell in cells}
        
        lin_pred_st_all = \
        {cell:[[] for _ in range(len(spikeTriggeredAnalysis.sta[cell]))] \
                                   for cell in cells}
        for cell in cells:
            
            
            n_bins = []
            min_bin_edge = np.inf
            max_bin_edge = -np.inf
            for n_partition, partition_sta in \
            enumerate(spikeTriggeredAnalysis.sta[cell]):
            
                p = spikeTriggeredAnalysis.spike_partition[n_partition]
                #get the training stimulus snippets
                temp1 = stimulus_snippets[:, :p[0]]
                temp2 = stimulus_snippets[:, p[1]:]
                partition_stim_snips = np.concatenate([temp1, temp2], axis=1)
                #filter response of raw stim ensemble 
                lin_pred_all = np.matmul(partition_sta, partition_stim_snips)
                linear_prediction[cell][n_partition] = lin_pred_all
                #filter response of STE
                lin_pred_st = np.matmul(partition_sta, \
                                spikeTriggeredAnalysis.ste[cell][n_partition])
                lin_pred_st_all[cell][n_partition] = lin_pred_st


                numbins = 8
                #determine bin edges from STE histogram with defined number of bins
                hist_st_prelim, bin_edges= \
                                    np.histogram(lin_pred_st, bins = numbins)
                binwidth = bin_edges[-1]-bin_edges[-2]
                min_stim_hist = np.min(lin_pred_all)
                max_stim_hist = np.max(lin_pred_all)
                if min_stim_hist<min_bin_edge:
                    min_bin_edge = min_stim_hist
                if max_stim_hist>max_bin_edge:
                    max_bin_edge = max_stim_hist
                
                lower_edge = bin_edges[0]
                upper_edge = bin_edges[-1]
                bin_edges_stim = []
                while lower_edge>min_stim_hist:
                    bin_edges_stim.insert(0, lower_edge - binwidth)
                    lower_edge = bin_edges_stim[0]
                [bin_edges_stim.append(bin_edges[i]) for i in range(len(bin_edges))]
                while upper_edge<max_stim_hist:
                    bin_edges_stim.append(upper_edge+binwidth)
                    upper_edge = bin_edges_stim[-1]
                    
                print('cell: {}, number bins : {}'.format(cell, len(bin_edges_stim)))
                n_bins.append(len(bin_edges_stim))
#                 hist_stim, _ = np.histogram(lin_pred_all, bin_edges_stim)
#                 hist_stim = hist_stim.astype(float)
#                 hist_stim[hist_stim ==0] += 1
#                 hist_ste = np.zeros_like(hist_stim, dtype=float)
#                 idx = np.where(bin_edges_stim==bin_edges[0])[0][0]
#                 hist_ste[idx:idx+len(hist_st_prelim)] = hist_st_prelim
#                 hist_ste = hist_ste.astype(float)
#                 binned_linear_prediction[cell][n_partition] = \
#                 bin_edges_stim[:-1]+np.diff(bin_edges_stim)/2
#                 
#                 binned_nonlinear_prediction[cell][n_partition] = \
#                 (hist_ste/hist_stim)
#                 hist_ste_all[cell][n_partition] = hist_ste
#                 hist_stim_all[cell][n_partition] = hist_stim
#                 hist_bin_edges_all[cell][n_partition] = bin_edges_stim

            med_n_bins = int(np.median(n_bins))
            for n_partition, partition_sta in \
            enumerate(spikeTriggeredAnalysis.sta[cell]):
             
             
                p = spikeTriggeredAnalysis.spike_partition[n_partition]
                #get the training stimulus snippets
                temp1 = stimulus_snippets[:, :p[0]]
                temp2 = stimulus_snippets[:, p[1]:]
                partition_stim_snips = np.concatenate([temp1, temp2], axis=1)
                lin_pred_all = np.matmul(partition_sta, partition_stim_snips)
                linear_prediction[cell][n_partition] = lin_pred_all
                lin_pred_st = np.matmul(partition_sta, \
                                spikeTriggeredAnalysis.ste[cell][n_partition])
                 
                hist_stim, bin_edges_stim = np.histogram(lin_pred_all, \
                                        med_n_bins, \
                                        range = (min_bin_edge, max_bin_edge))
                hist_stim = hist_stim.astype(float)
                hist_stim[hist_stim ==0] += 0.0001
                hist_ste, _ = np.histogram(lin_pred_st, bin_edges_stim)
                 
                binned_linear_prediction[cell][n_partition] = \
                bin_edges_stim[:-1]+np.diff(bin_edges_stim)/2
                 
                binned_nonlinear_prediction[cell][n_partition] = \
                (hist_ste/hist_stim)
                hist_ste_all[cell][n_partition] = hist_ste
                hist_stim_all[cell][n_partition] = hist_stim
                hist_bin_edges_all[cell][n_partition] = bin_edges_stim
                
            
        self.hist_bin_edges = hist_bin_edges_all
        self.hist_ste = hist_ste_all
        self.hist_stim = hist_stim_all  
        self.linear_prediction = linear_prediction
        self.binned_linear_prediction = binned_linear_prediction
        self.binned_nonlinear_prediction = binned_nonlinear_prediction
                



    def fit_nonlinear_function(self, normed = True):
        '''
        This method loops over all given cells and fits the nonlinear function
        that best describes the relationship between the binned linear prediction
        and the binned nonlinear prediction
        Currently it calls fit_sigmoid() and fit_exponential() and receives 
        popt (optimal parameters) and r (r_squared) for both nonlinear functions
        and saves these in a dictionary with the cell identities
        as keys. popt and r are lists with n_partition entries
        
        Inputs:
        -----------------------------------------------------------------------
        normed    boolean
                  whether the nonlinearity should be fit the the normed 
                  binned_linear_prediction and binned_nonlinear_prediction, i.e.
                  divided by n_repetitions
                  
        Outputs:
        -----------------------------------------------------------------------
        None;
        sets
        self.fit_normed_nonlinearity    boolean, set to the value of normed
        self.popt_sig_all
        self.r_sig_all
        self.popt_exp_all
        self.r_exp_all
        self.success_sig_all
        self.success_exp_all
        
        are dictionaries with cell IDs as keys and the outputs from fit_sigmoid
        and fit_exponential, respectively, as values 
        '''
        
        self.fit_normed_nonlinearity = normed
        try:
            self.binned_linear_prediction
        except(AttributeError):
            print('You have to create a linear prediction first!')

        popt_sig_all = {cell:[] for cell in self.cells}
        r_sig_all = {cell:[] for cell in self.cells}
        success_sig_all = {cell:[] for cell in self.cells}
        popt_exp_all = {cell:[] for cell in self.cells}
        r_exp_all = {cell:[] for cell in self.cells}
        success_exp_all = {cell:[] for cell in self.cells}
        for cell in self.cells:
            
            popt_sig, r_sig, success_sig = self.fit_sigmoid(cell)
            popt_sig_all[cell] = popt_sig
            r_sig_all[cell] = r_sig
            success_sig_all[cell] = success_sig
            popt_exp, r_exp, success_exp = self.fit_exponential(cell)
            popt_exp_all[cell] = popt_exp
            r_exp_all[cell] = r_exp
            success_exp_all[cell] = success_exp
        
        self.popt_sig_all = popt_sig_all
        self.r_sig_all = r_sig_all
        self.popt_exp_all = popt_exp_all
        self.r_exp_all = r_exp_all
        self.success_sig_all = success_sig_all
        self.success_exp_all = success_exp_all
            
            
        
            
            

    
    def fit_sigmoid(self, cell):
        '''
        This method operates on single cells. It loops over n_partition 
        different binned linear and corresponding binned nonlinear predictions
        and tries to fit a sigmoid to them using curve_fit (scipy.optimize.curve_fit)
        It also evaluates the goodness of fit using the method test_goodness_of_fit
        Dependencies: sigmoid(), test_goodness_of_fit, imported from analysis.utils
         

        Output:
        
        popt:    list of length n_partition, each entry containing the optimal
                 parameters for that partition
        r        list of length n_partition, each entry containing the goodness
                 of fit of that  partition
        success:    list
                    of length n_partition, each entry containing a boolean value
                    indicating whether fitting was successful for this cell and 
                    data from this partition
        '''
        
        blp = self.binned_linear_prediction[cell]
        y = self.binned_nonlinear_prediction[cell]
        if self.fit_normed_nonlinearity:
            n_partitions = len(y)
            y = [y_/(n_partitions-1) for y_ in y]
        popt = [[] for _ in range(len(blp))]
        r = [[] for _ in range(len(blp))]
        success = [[] for _ in range(len(blp))]
    
        for i, x in enumerate(blp):
            y_max = max(y[i])
            y_min = min(y[i])
            x_max = x[np.argmax(y[i])]
            x_min = x[np.argmin(y[i])]
            
            x0 = x_min + (x_max - x_min)/2
            '''
            parameters are 50 % threshold, maximum y value, minimum y value and gain;
            initialize as: 
            '''
            p0 = [x0, y_max, y_min, 1]
            try:
                popt[i], _ = curve_fit(sigmoid, x, y[i], p0, maxfev = 2000)
                y_est = sigmoid(x, *popt[i])
                r[i] = test_goodness_of_fit(y[i], y_est)
                success[i] = True
            except(RuntimeError):
                success[i] = False
                popt[i] = []
                r[i] = 0
            

        return popt, r, success
    
    
    def fit_exponential(self, cell):
        '''
        This method operates on single cells. It loops over n_partition 
        different binned linear and corresponding binned nonlinear predictions
        and tries to fit an exponential to them using curve_fit (scipy.optimize.curve_fit)
        It also evaluates the goodness of fit using the method test_goodness_of_fit
        Dependencies: exponential(), test_goodness_of_fit, imported from analysis.utils
         
        Output:
        
        popt:    list of length n_partition, each entry containing the optimal
                 parameters for that partition
        r        list of length n_partition, each entry containing the goodness
                 of fit of that  partition
        success:    list
                    of length n_partition, each entry containing a boolean value
                    indicating whether fitting was successful for this cell and 
                    data from this partition
        '''

        blp = self.binned_linear_prediction[cell]
        y = self.binned_nonlinear_prediction[cell]
        if self.fit_normed_nonlinearity:
            n_partitions = len(y)
            y = [y_/(n_partitions-1) for y_ in y]
        popt = [[] for _ in range(len(blp))]
        r = [[] for _ in range(len(blp))]
        success = [[] for _ in range(len(blp))]

    
        for i, x in enumerate(blp):


            y_max = max(y[i])
            y_min = min(y[i])
            x_max = x[np.argmax(y[i])]
            x_min = x[np.argmin(y[i])]
            multiplicative_factor = (y_max-y_min)*0.02
            gain = 1
            offset = y_min
            p0 = [multiplicative_factor, gain, offset]
            try:
                popt[i], _ = curve_fit(exponential, x, y[i], p0, maxfev = 2000)
                y_est = exponential(x, *popt[i])
                r[i] = test_goodness_of_fit(y[i], y_est)
                success[i] = True
            except(RuntimeError):
                success[i] = False
                popt[i] = []
                r[i] = 0

            

        return popt, r, success 
    
    def plot_nonlinearity(self, sta, path, cells = None):
        '''
        
        '''
        
        if cells == None:
            cells = self.cells
        
        #determine the number of subplot 
        n_subplots_per_figure= 16 
        n_fig = 0
        quality_of_fit = [[] for n in range(sta.n_clusters)]
        for i, cell in enumerate(cells):
            
            cluster_ = sta.cluster_labels[cell]
            cl = sta.color_mapping[sta.colors_dict[cell]][0]
        
            if i%n_subplots_per_figure ==0:
                if i>0:
                    plt.savefig(path+'nonlinearity_'+str(n_fig)+'.png', fmt = 'png', dpi = 300)
                    plt.close()
                n_fig+=1
                plt.figure(n_fig, figsize=(16,12))

            
            plt.subplot(np.sqrt(n_subplots_per_figure),np.sqrt(n_subplots_per_figure),\
                        i%n_subplots_per_figure+1)
            plt.title('cell {}'.format(cell))
            for n_partition, partition in \
            enumerate(self.binned_linear_prediction[cell]):
                plt.scatter(partition, \
                            self.binned_nonlinear_prediction[cell][n_partition]/4, color = cl)
                
                r_sig = self.r_sig_all[cell][n_partition]
                r_exp = self.r_exp_all[cell][n_partition]
                
                if r_sig>r_exp:
                    
                    quality_of_fit[cluster_].insert(0, r_sig)
                    if self.success_sig_all[cell][n_partition] & (r_sig>0.7):
                        sig_pred = sigmoid(partition, *self.popt_sig_all[cell][n_partition])
                        plt.plot(partition, sig_pred, linestyle = '--', color = cl, label = 'r_sig = '+str(r_sig))
#                 else:
#                     print('Sigmoid fit was not successfull for cell {}, partition {} (r_sig = {}) '.format(cell, n_partition, r_sig))
#                     
                else:
                    quality_of_fit[cluster_].insert(0, r_sig)
                    if self.success_exp_all[cell][n_partition] & (r_exp>0.7):
                        exp_pred = exponential(partition, *self.popt_exp_all[cell][n_partition])
                        plt.plot(partition, exp_pred, color = cl, label = 'r_exp = '+str(r_exp))
#                 else:
#                     print('Exp. fit was not successfull for cell {}, partition {} (r_sig = {}) '.format(cell, n_partition, r_exp))
        
                  
        plt.savefig(path+'nonlinearity_'+str(n_fig)+'.png', fmt = 'png', dpi = 300)   
        
        for i in range(sta.n_clusters):
            print('cluster {}: mean : {}, std: {}'.format(i, np.average(quality_of_fit[i]), np.std(quality_of_fit[i])))
    
    
    def predict(self, cells, spikeTriggeredAnalysis): 
        '''
        Computes linear and nonlinear model prediction of the cells' firing rate
        in response to the stimulus; the version of the model fit on a training 
        set, e.g. the spikes in response to the stimulus between 1000 and 5000
        ms, predicts the leftout dataset, e.g. in this case spikes in response
        to the stimulus between 0 and 1000 ms. Because the model takes a certain
        fraction of the stimulus for prediction (e.g., 20 ms), the first predicted
        value is for the t=20ms. Therefore, the prediction is shifted forward
        by this amount.
        
        CAVEAT: THIS FUNCTION ONLY WORKS PROPERLY FOR PREDICTING RGC SPIKES WITH
        MILLISECOND RESOLUTION 
        
        Inputs:
        -----------------------------------------------------------------------
        cells:    list
                  list of cell IDs
        spikeTriggeredAnalysis    instantiated object of the SpikeTriggeredAverage
                                  class
                                  
        Outputs:
        -----------------------------------------------------------------------
        None;
        sets
        self.roll_factor    int    
                            the amount (in ms) by which the prediction has to be
                            (and has been) shifted 
        
        self.lin_prediction     dictionary
        self.nonlin_prediction  dictionary
                                with cell IDs as keys and a list with n_partition
                                elements as value where each element contains the
                                model's linear or nonlinear (respectively) 
                                prediction of the test stimulus snippets in that
                                partition
        self.test_stim_snips    list
                                with n_partition elements, each element containing
                                the test stimulus snippets for that partition
                            
        
        
        '''
        
        stimulus_snippets = spikeTriggeredAnalysis.used_stimulus_ensemble
        test_stim_snips = [stimulus_snippets[:, p[0]:p[1]] for p in\
                               spikeTriggeredAnalysis.spike_partition]
        
        
        
        roll_factor = spikeTriggeredAnalysis.delta_t_to_spike_back
        self.roll_factor = roll_factor
        
        lin_prediction = {cell: [np.zeros(test_stim_snip.shape[-1])\
                   for test_stim_snip in test_stim_snips] for cell in cells}
        
        
        nonlin_prediction = {cell: [np.zeros(test_stim_snip.shape[-1])\
                   for test_stim_snip in test_stim_snips] for cell in cells}
        
        for cell in cells:
            for n, test_stim_snip in enumerate(test_stim_snips):
                if self.success_exp_all[cell][n] and self.success_sig_all[cell][n]:
                    if self.r_exp_all[cell][n]>self.r_sig_all[cell][n]:
                        popts = self.popt_exp_all[cell][n]
                        lin_prediction[cell][n] = \
                        np.matmul(spikeTriggeredAnalysis.sta[cell][n], test_stim_snip)
                        lin_prediction[cell][n] = np.roll(lin_prediction[cell][n], roll_factor)
                        lin_prediction[cell][n][:roll_factor] = 0
                        nonlin_prediction[cell][n] = exponential(lin_prediction[cell][n], *popts)
                        
                    else:
                        popts = self.popt_sig_all[cell][n]
                        lin_prediction[cell][n] = \
                        np.matmul(spikeTriggeredAnalysis.sta[cell][n], test_stim_snip)
                        lin_prediction[cell][n] = np.roll(lin_prediction[cell][n], roll_factor)
                        lin_prediction[cell][n][:roll_factor] = 0
                        nonlin_prediction[cell][n] = sigmoid(lin_prediction[cell][n], *popts)
                elif self.success_exp_all[cell][n]:
                    popts = self.popt_exp_all[cell][n]
                    lin_prediction[cell][n] = \
                        np.matmul(spikeTriggeredAnalysis.sta[cell][n], test_stim_snip)
                    lin_prediction[cell][n] = np.roll(lin_prediction[cell][n], roll_factor)
                    lin_prediction[cell][n][:roll_factor] = 0
                    nonlin_prediction[cell][n] = exponential(lin_prediction[cell][n], *popts)
                elif self.success_sig_all[cell][n]:
                    popts = self.popt_sig_all[cell][n]
                    lin_prediction[cell][n] = \
                        np.matmul(spikeTriggeredAnalysis.sta[cell][n], test_stim_snip)
                    lin_prediction[cell][n] = np.roll(lin_prediction[cell][n], roll_factor)
                    lin_prediction[cell][n][:roll_factor] = 0
                    nonlin_prediction[cell][n] = sigmoid(lin_prediction[cell][n], *popts)
                else:
                    print('cell no {} could not be fitted successfully; no full model prediction available'.format(cell))
        self.lin_prediction = lin_prediction
        self.nonlin_prediction = nonlin_prediction  
        self.test_stim_snips = test_stim_snips     
    
    
    def evaluate(self, cells, spikes, plot = False):
        '''
        Evaluates the prediction performance of the model after the linear and 
        after linear+nonlinear stage. It computes the correlation coefficient 
        between the true binned firing rate and the prediction calculated in 
        predict()
        
        Inputs:
        -----------------------------------------------------------------------
        cells    list
                 of cell IDs
        spikes    instantiated object of class Spikes()
        
        Outputs:
        -----------------------------------------------------------------------
        None;
        sets
        self.true_firing_rate    dictionary    
                                 with cell IDs as keys and a list of length
                                 n_partition, where each entry contains the 
                                 (normed) binned firing rate of the cell in that
                                 partition 
                                 
        self.lin_corr
        self.nonlin_corr        dictionaries 
                                with cell IDs as keys and a list of length 
                                n_partition, where each entry contains the rounded
                                correlation coefficient between true firing rate
                                and model prediction for that partition
        
        
        '''
        n_bins = 1001 # this should be a parameter that is passed, so that the
                        #performance of the model at different resolutions
                        #can be evaluated 
        n_partitions = len(spikes.partition)
        lin_corr = {cell:[[] for _ in range(n_partitions)] for cell in cells}
        nonlin_corr = {cell:[[] for _ in range(n_partitions)] for cell in cells}
        self.true_firing_rate = {cell:[[] for _ in range(n_partitions)] for cell in cells}
        for cell in cells:
            for i, partition in enumerate(spikes.partition):
                test_spikes = np.concatenate(spikes.test_set[cell][i])
                pred = self.lin_prediction[cell][i]
                bins = np.linspace(partition[0], partition[0]+len(pred), num = pred.shape[0]+1)
                true,_ = np.histogram(test_spikes, bins=bins)
                if self.fit_normed_nonlinearity:
                    true = true/n_partitions
                self.true_firing_rate[cell][i] = true
                
                if pred.shape[0]!= true.shape[0]:
                    raise ValueError()
                

#                 lin_corr[cell][i] = \
#                 np.round(np.corrcoef(true[:pred.shape[0]], pred, \
#                                      rowvar=True)[0][1], 2)
                lin_corr[cell][i] = \
                np.round(np.corrcoef(true, pred, \
                                     rowvar=True)[0][1], 2)

                pred = self.nonlin_prediction[cell][i]
#                 nonlin_corr[cell][i] = \
#                 np.round(np.corrcoef(true[:pred.shape[0]], pred)[0][1], 2)
                nonlin_corr[cell][i] = \
                    np.round(np.corrcoef(true, pred)[0][1], 2)
            
           
        
        self.lin_corr = lin_corr
        self.nonlin_corr = nonlin_corr
        
#         if plot:
#             plt.figure()
#             plt.subplot(2,1,1)
#             plt.hist(lin_corr, bins = 20)
#             plt.hist(nonlin_corr, bins = 20)
            
                
    
    def plot_this(self, cells, stimulus, spikes, sta, \
                  plot_style = 'seaborn-paper',bias = None, path = None, \
                  fname = None):
        '''
        Plot the true and predicted firing rates for the cells
        '''
        plt.style.use(plot_style)
        fig = plt.figure(figsize = (12,9))
        n_rows = len(cells) + 1
        n_cols = 1
        multiplier = 1
        filter_pos = 0
        nonlin_pos = 0
        raster_pos = 0
        if False:
#         if False:
            cell_biases = [b for cell, b in bias.items() if cells.count(cell)]
            plot_position = np.argsort(cell_biases)
        else:
            if hasattr(sta, 'cluster_labels'):
                cell_cluster_labels = \
                [label for cell, label \
                    in sta.cluster_labels.items()\
                    if cells.count(cell)]
                plot_position = np.argsort(cell_cluster_labels)
            else:
                plot_position = np.linspace(0, len(cells)-1,len(cells))
        

        
        '''
        top middle
        #######################################################################
        plot e-stim excerpt (1000ms)
        '''
        ax_ref_psth = plt.subplot2grid([n_rows, n_cols], \
                                       [0,raster_pos*multiplier], colspan=1)
        x_el_stim = stimulus.time_line[:10000]*10**-6 #convert from us to s
        y_el_stim = stimulus.current_density[:10000]
        
        plt.plot(x_el_stim, y_el_stim, color = 'k', label = 'el. stimulus', \
                 linewidth = 0.7)
        transform_obj = ax_ref_psth.transData
        horizontal_scalebar = AnchoredSizeBar(transform_obj, 0.1,\
                           '0.1 s', 'upper center', 
                           borderpad=-1,
                           frameon=False,
                           label_top=True
                           )
        vertical_scalebar = AnchoredSizeBar(transform_obj, 0.001,\
                           r'1 $\frac{mA}{cm^2}$', 'center right',
                           frameon=False,borderpad=-1, label_top=True,
                           size_vertical=1
                           )
        ax_ref_psth.add_artist(horizontal_scalebar)
        ax_ref_psth.add_artist(vertical_scalebar)

         
         
        
#    
        for i, cell in enumerate(cells):
            
            
            '''
            ###################################################################
            plot STA
            '''
#             pp = np.where(plot_position == i)[0][0]
#             ax = plt.subplot2grid([n_rows, n_cols], [pp+1, filter_pos])
#             
#             x = np.linspace(-0.02,0.01,sta.len_sta_samples)    
#             sta_temp = np.average(sta.sta[cell], 0)
#             normed_sta = sta_temp/np.max(sta_temp)
#             cl_LNP = sta.color_mapping[sta.colors_dict[cell]][0]
#             plt.plot(x, normed_sta, color = cl_LNP)
# #   
#             (ymin, ymax) = ax.get_ylim()
# 
#             plt.vlines(0,0, 1, color = 'k')
#             if i == 0:
#                 plt.annotate(r'1 $\frac{mA}{cm^2}$', (-0.006, 1))
#             for label in ax.get_xticklabels():
#                 label.set_visible(False)
#    
#             if hasattr(sta, 'cluster_labels') and False:
#                 plt.ylabel('id: {} ,b: {}, cl: {}'.format(cell, cell_biases[i], \
#                 sta.cluster_labels[cell]) ,rotation = 'horizontal', size='medium')
#             elif hasattr(sta, 'cluster_labels'):
#                 plt.ylabel('id:{}, cl: {}'.format(cell, \
#                                 sta.cluster_labels[cell]) , \
#                                                  rotation = 'horizontal', size ='medium')
#             else:
#                 plt.ylabel('id:{}'.format(cell),size='medium')
#             
#             ax.yaxis.labelpad = 75  
            '''
            middle left
            ###################################################################
            plot nonlinearity for all cells
            '''
            cl_LNP = sta.color_mapping[sta.colors_dict[cell]][0]
            pp = np.where(plot_position == i)[0][0]
#             ax = plt.subplot2grid([n_rows, n_cols], [pp+1, nonlin_pos], colspan =1)
#             preds = []
#             for n_partition, partition in \
#             enumerate(self.binned_linear_prediction[cell]):
# #                 ax.scatter(partition, \
# #                             self.binned_nonlinear_prediction[cell][n_partition], marker='*')
#                 
#                 r_sig = self.r_sig_all[cell][n_partition]
#                 r_exp = self.r_exp_all[cell][n_partition]
#                 if r_sig>r_exp:
#             
#                     if self.success_sig_all[cell][n_partition] & (r_sig>0.7):
#                         sig_pred = sigmoid(partition,\
#                                          *self.popt_sig_all[cell][n_partition])
# 
#                         plt.plot(partition, sig_pred, color = 'k',\
#                                   label = 'r_sig = '+str(r_sig))
#                         preds.append(sig_pred)
# 
#                 else:
#                     if self.success_exp_all[cell][n_partition] & (r_exp>0.7):
#                         exp_pred = exponential(partition, \
#                                         *self.popt_exp_all[cell][n_partition])
# 
#                         plt.plot(partition, exp_pred, color = 'darkgrey', \
#                                  label = 'r_exp = '+str(r_exp))
#                         
#                         preds.append(exp_pred)
#             if len(preds)>0:
#                 preds = np.concatenate(preds)
#                 ymax = np.round(np.max(preds),2)
#                 plt.annotate(r'$r_{max}$:'+str(ymax), (0.8, 0.5),\
#                               xycoords = 'axes fraction')
#             plt.ylabel('{}'.format(cell), rotation = 'horizontal')
#             ax.yaxis.labelpad = 75
            
            '''
            middle middle
            ###################################################################
            plot real rate 

            '''
            ax = plt.subplot2grid([n_rows, n_cols], [pp+1, raster_pos],\
                                   colspan = 1)
            bin_n = 1000
            x_limit = 1
            factor = 1000*x_limit/(bin_n-1)
            bins = np.linspace(0, x_limit, bin_n)
            spikes.plot_psth(cell, bins,  ax = ax, normed = True,\
                                        ms=True)

            '''
            middle middle
            ###################################################################
            plot predicted rate 

            '''
# 
            x = np.linspace(0, 1, 1000)

            ax.plot(x[self.roll_factor:], factor*self.nonlin_prediction[cell][0]\
                    [self.roll_factor:], color = cl_LNP, alpha = 0.5, \
                    label = self.nonlin_corr[cell][0])
#             ax.legend(loc = 'upper right', fontsize = 'small')  
#             plt.annotate('avg. corr. : {}'.format(self.nonlin_corr[cell][0]),\
#                          (1,0.5),xycoords = 'axes fraction')
           
            plt.ylabel('{}: corr= {}'.format(cell, \
                np.average(self.nonlin_corr[cell])), rotation = 'horizontal')

        for ax in fig.get_axes():
            for label in ax.get_xticklabels():
                label.set_visible(False)
            for label in ax.get_yticklabels():
                label.set_visible(False)
        
            ax.tick_params(axis = 'both', which ='both', bottom = False,\
                            left = False, right = False, top = False)
            ax.spines["top"].set_visible(False)    
            ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(False) 
            ax.spines["bottom"].set_visible(False)
            
        
        
        if path is not None:
            if fname is None:
                fname = 'LNP_prediction'
            else:
                fname = 'LNP_prediction'+fname
#             plt.savefig(path+fname+'.ps', dpi = 100, fmt='ps')
            plt.savefig(path+fname+'.png', dpi = 300, fmt='png')
#             plt.savefig(path+fname+'.svg', dpi = 100, fmt='svg')

            plt.close(fig)

            
        
        
        