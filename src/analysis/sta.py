'''
Created on 13.08.2018

@author: larissa.hoefling@uni-tuebingen.de
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster, set_link_color_palette
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from copy import deepcopy
from time import time 
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pickle as pkl
import pandas as pd
from _operator import index

class SpikeTriggeredAnalysis():
    
    '''
    Objects of the class SpikeTriggeredAnalysis compute the spike-triggered 
    average of all or some cells from one recording relative to a provided stimulus.  
    
    attributes:
    ###########################################################################
    len_sta_samples    int
                       length of the sta in stimulus samples
    len_sta_ms          int
                        length of the sta in ms 
    robust             boolean
                        whether the sta is computed on robust spikes (see spikes
                        documentation)
    whitening          boolean
                        whether the sta is computed on the whitened stimulus
    delta_t_to_spike_back     int
                              how far back into the past the sta is computed
                              relative to the spike (in ms)
    delta_t_to_spike_forward  int
                              how far into the future relative to the spike the 
                              sta is computed (in ms)
                              
    used_stimulus_ensemble    array of dimensions len_sta x # of snippets
                              the stimulus ensemble that is used for computing 
                              the sta
        mode            string
                        full or cv (cross validated)
    spike_partition    see spikes.py
    
    ste, sta, ste_stand_dev, sta_norm, indices: see compute_spike_triggered_average
    
    methods:
    ###########################################################################
    get_spike_triggered_ensemble(spikes, stimulus, delta_t_to_spike_back, delta_t_to_spike_forward,
                                cells, whitening, rcond)
                                
    '''


    def __init__(self, cells):

        self.len_sta_samples = np.nan
        self.len_sta_ms = np.nan
        self.color_mapping = {'k' : ['k']}
        self.cluster_labels = {cell: -1 for cell in cells}
        self.colors_dict = {cell:'k' for cell in cells}
                
    def get_spike_triggered_ensemble(self, spikes, stimulus, mode = 'full', \
                                     delta_t_to_spike_back = 20, 
                                     delta_t_to_spike_forward = 0, cells = None,   
                                     whitening = False, robust = False):
        '''
        computes the spike triggered ensemble, the spike-triggered average and 
        the standard deviation of the STE of the specified cells, and saves them
        as instance variables self.ste, self.sta and self.ste_stand_dev. These
        instance variables are dictionaries, with the cell numbers (starting at
        1) as keys, and each key is pointing to a list with [number of partition]
        entries containing the ste, sta or std, respectively, of different stimulus 
        repetitions. It converts the spikes to indices into the stimulus ensemble
        (i.e. for spike at time t, take the stimulus snippet at 
        t- delta_t_to_spike_back), and also saves these indices as instance variable.
        computes the L1 norm of the STA
        
        Inputs:
        -----------------------------------------------------------------------
        spikes    an instance of the class Spikes()
        stimulus    an instance of the class Stimulus()
        mode    string 
                full: compute the STA on all spikes
                cv: compute the STA on the n_partition training sets 
        delta_t_to_spike_back    int
                                 the pre spike time, in ms
        delta_t_to_spike_forward    int    
                                    the post spike time, in ms
        cells       list
                    if None, perform for all cells in spikes
        whitening    boolean
                     whether to compute the STA on the whitened stimulus
        robust        boolean
                    whether to compute the STA on the robust spikes
        
        Outputs:
        -----------------------------------------------------------------------
        None; sets 
        self.len_sta_samples 
        self.len_sta_ms 
        self.robust
        self.whitening 
        self.delta_t_to_spike_back 
        self.delta_t_to_spike_forward 
        self.used_stimulus_ensemble
        self.ste
        self.sta    list
                    of length n_partitions, each containing the STA computed for 
                    the training set of spikes of this partition, shape 
                    (len_sta_samples,)
        self.ste_stand_dev
        self.sta_norm
        self.indices
        self.spike_partition
        self.mode
        '''
        
        
        factor = int(stimulus.temp_res/1000) # the factor to convert from ms (spikes) to samples (0.1ms, stimulus)
        len_sta_ms = delta_t_to_spike_back-delta_t_to_spike_forward
        len_sta_samples = len_sta_ms*factor
        self.len_sta_samples = len_sta_samples
        self.len_sta_ms = len_sta_ms
        self.robust = robust
        self.whitening = whitening
        self.delta_t_to_spike_back = delta_t_to_spike_back
        self.delta_t_to_spike_forward = delta_t_to_spike_forward
        
        if whitening:
            stim_ensemble = \
            stimulus.create_stimulus_ensemble(self.len_sta_samples, \
                                              stepwidth=1, white=True)
        else:
            stim_ensemble = \
            stimulus.create_stimulus_ensemble(self.len_sta_samples, \
                                              stepwidth=1, white=False)
        self.used_stimulus_ensemble = stim_ensemble

        
        if cells == None:
            cells = [cell for cell in spikes.structured_ts.keys()]
        
        if robust:
            if mode == 'full':
                print('The full STA is computed')
                time_stamps = spikes.robust_spikes
                
            elif mode == 'cv':
                print('The cv STA version is computed')
                time_stamps = {cell:[] for cell in cells}
                time_stamps_all = spikes.robust_training_set
                for cell in cells:
                    for partition_spikes in time_stamps_all[cell]:
                        time_stamps[cell].append(np.concatenate(partition_spikes))
                self.spike_partition = spikes.partition
            else:
                print('No valid mode was passed!')
                
        else:
            if mode == 'full':
                print('The full STA is computed')
                time_stamps = spikes.structured_ts
                
            elif mode == 'cv':
                print('The cv STA version is computed')
                time_stamps = {cell:[] for cell in cells}
                time_stamps_all = spikes.training_set
                for cell in cells:
                    for partition_spikes in time_stamps_all[cell]:
                        #append spikes from all repetitions within one partition
                        time_stamps[cell].append(np.concatenate(partition_spikes))
                self.spike_partition = spikes.partition
            else:
                print('No valid mode was passed!')
            
        self.mode = mode
        

        self.ste = \
        {cell:[[] for _ in range(len(time_stamps[cell]))] for cell in cells}
        self.sta = \
        {cell:[[] for _ in range(len(time_stamps[cell]))] for cell in cells}
        self.ste_stand_dev = \
        {cell:[[] for _ in range(len(time_stamps[cell]))] for cell in cells}
        self.sta_norm = \
        {cell: [None for _ in range(len(time_stamps[cell]))] for cell in cells}
        self.indices = \
        {cell:[[] for _ in range(len(time_stamps[cell]))] for cell in cells}
        self.colors_dict = \
        {cell : 'k' for cell in cells}

        for cell in cells:
            for i, spikes_ in enumerate(time_stamps[cell]): 

                cond = np.logical_and(spikes_>delta_t_to_spike_back, \
                                      spikes_<(stim_ensemble.shape[1]))
                spikes_ = spikes_[cond]
                indices = np.round(spikes_).astype(int) - delta_t_to_spike_back
                self.ste[cell][i] = stim_ensemble[:, indices]
                self.sta[cell][i] = np.average(self.ste[cell][i], 1)
                self.ste_stand_dev[cell][i]=np.std(self.ste[cell][i], 1)
                self.sta_norm[cell][i] = np.linalg.norm(self.sta[cell][i])
                self.indices[cell][i] = indices
                
                
        time_stamps = spikes.structured_ts
        self.full_ste = \
        {cell:[[] for _ in range(len(time_stamps[cell]))] for cell in cells}
        self.full_sta = \
        {cell:[[] for _ in range(len(time_stamps[cell]))] for cell in cells}
        self.full_ste_stand_dev = \
        {cell:[[] for _ in range(len(time_stamps[cell]))] for cell in cells}
        self.full_sta_norm = \
        {cell: [None for _ in range(len(time_stamps[cell]))] for cell in cells}


        for cell in cells:
            for i, spikes in enumerate(time_stamps[cell]): 

                cond = np.logical_and(spikes>delta_t_to_spike_back, \
                                      spikes<(stim_ensemble.shape[1]))
                spikes = spikes[cond]
                indices = np.round(spikes).astype(int) - delta_t_to_spike_back
                self.full_ste[cell][i] = stim_ensemble[:, indices]
                self.full_sta[cell][i] = np.average(self.ste[cell][i], 1)
                self.full_ste_stand_dev[cell][i]=np.std(self.ste[cell][i], 1)
                
                
                

    def get_dummy_spike_triggered_ensemble(self, spikes, stimulus, \
                                     delta_t_to_spike_back = 20, 
                                     delta_t_to_spike_forward = 0, cells = None,   
                                     whitening = False, bootstrap_reps = 100,
                                     robust = False):
        '''
        same as get_spike_triggered_ensemble, but decorrelates the stimulus 
        snippets from the spikes by randomly shuffling them before computing the 
        "STA"
        
        Inputs:
        -----------------------------------------------------------------------
        as get_spike_triggered_ensemble
        additionally
        bootstrap_reps    int
                          how many times the dummy sta is computed
        Outputs:
        -----------------------------------------------------------------------
        None; sets
        self.sta_norm_distribution    a dictionary with cell IDs as keys and 
                                      lists of size bootstrap_reps for each cell
                                      containing the sta norms for every dummy
                                      sta (yielding the sta norm null distribution)

        '''
         
        factor = int(stimulus.temp_res/1000) # the factor to convert from ms (spikes) to samples (0.1ms, stimulus)
        len_sta_ms = delta_t_to_spike_back-delta_t_to_spike_forward
        len_sta_samples = len_sta_ms*factor
         
        if whitening:
            stim_ensemble = \
            stimulus.create_stimulus_ensemble(len_sta_samples, \
                                              stepwidth=1, white=True)
            transposed_ensemble = np.transpose(stim_ensemble)

        else:
            stim_ensemble = \
            stimulus.create_stimulus_ensemble(len_sta_samples, \
                                              stepwidth=1, white=False)
            transposed_ensemble = np.transpose(stim_ensemble)

         
        if cells == None:
            cells = [cell for cell in spikes.structured_ts.keys()]
            
        if robust:
            time_stamps = spikes.robust_spikes
        else:
            time_stamps = spikes.structured_ts
           
        ste = \
        {cell:[[] for _ in range(bootstrap_reps)] for cell in cells} 
        
        sta = \
        {cell:[[] for _ in range(bootstrap_reps)] for cell in cells}
        
        sta_norm = \
        {cell:[[] for _ in range(bootstrap_reps)] for cell in cells}
        
        self.sta_norm_dist = {cell: [] for cell in cells}
        
        self.sta_norm_percentage = {cell: [] for cell in cells}
        
         
        for cell in cells:
            
            spikes = np.concatenate(time_stamps[cell])

            cond = np.logical_and(spikes>delta_t_to_spike_back, \
                                      spikes<(stim_ensemble.shape[1])+\
                                      delta_t_to_spike_forward)

            spikes = spikes[cond]
            indices = np.round(spikes).astype(int) - delta_t_to_spike_back
            for i in range(bootstrap_reps):
                np.random.shuffle(transposed_ensemble)
                stim_ensemble = np.transpose(transposed_ensemble)
                ste = stim_ensemble[:, indices]
                sta[cell][i] = np.average(ste, 1)
                sta_norm[cell][i] = linalg.norm(sta[cell][i])
            self.sta_norm_dist[cell]= np.asarray(sta_norm[cell])
            percentage_larger_than = np.sum(sta_norm[cell]>self.sta_norm[cell])/bootstrap_reps
            self.sta_norm_percentage[cell] = percentage_larger_than
            
            
    def align_stas_to_minimum(self, cells):
        
        aligned_sta = \
        {cell:[[] for _ in range(len(self.sta[cell]))] for cell in cells}
        
        min_peak_pos = \
        {cell:[[] for _ in range(len(self.sta[cell]))] for cell in cells}
        
        latest_peak = 0
        earliest_peak = self.len_sta_samples
        for cell in cells:
            for i, sta in enumerate(self.sta[cell]):
                temp = np.argmin(sta)
                if temp>latest_peak:
                    latest_peak = temp
                if temp<earliest_peak:
                    earliest_peak = temp
                min_peak_pos[cell][i] = temp
                
        to_shift = latest_peak - earliest_peak
        
        for cell in cells:
            
            aligned_sta[cell] = [np.roll(sta,-to_shift)[:-to_shift] for sta in self.sta[cell]]
        
        self.aligned_sta = aligned_sta 
                
        
        
        
        
        
        
        

#     def cluster_sta(self, cells, n_clusters=2,  pca = True):
#         
#         
#         #determine the cells with non NaN STA
# 
#         X_input = [np.transpose(value) for value in self.sta.values()]
#         X_input = np.average(np.asarray(X_input), axis = 2)
#         
#         self.cluster_labels = {cell:-1 for cell in cells}
#         new_cells = deepcopy(cells)
#         boolean_mask = np.ones(X_input.shape[0], dtype=bool)
#         for cell_no, cell_id in enumerate(cells):
#             if np.any(np.isnan(X_input[cell_no])):
#                 new_cells.remove(cell_id)
#                 boolean_mask[cell_no] = False
#                 print(cell_id)
#         
#         X_input = X_input[boolean_mask, :]
#         if pca:
#             my_pca = PCA(n_components=0.9, svd_solver='full')
#             
#             X_output = my_pca.fit_transform(X_input)
#             print('Number of components : {}'.format(my_pca.n_components_))
# 
#         my_kmeans = KMeans(n_clusters)
#         labels = my_kmeans.fit_predict(X_output)
#         colors = ['r' if labels[i] == 0 else 'k' for i in range(len(labels))]
#         plt.figure()
#         for i, cell in enumerate(new_cells):
#              
#             self.cluster_labels[cell] = labels[i] 
#             plt.scatter(X_output[i, 0],X_output[i, 1], color = colors[i])
         
#         plt.scatter(my_kmeans.cluster_centers[])
#         n_clusters = 2
#         cluster_object = AgglomerativeClustering(n_clusters=n_clusters)
            
            
#         self.n_clusters = n_clusters

    def hierarchical_cluster(self, cells, n_clusters, marker_code = None, \
                              n_components = 0.9, use_pca_output = True,\
                               path = None, fname_prefix='default', \
                               metric = 'euclidean', method = 'average', \
                               pcs = [0,1],
                               pkl_path = None, \
                               mode = 'load', additional_STA_object = None,\
                               annotate_cell = None, znorm_this=False,
                               color_order = [0, 1, 2, 3, 4],
                             alternative_color_scheme=False,
                             ):
        '''
        Performs hierarchical clustering on the STAs from all passed cells. 
        If pca is True, perform hierarchical clustering in PC space.
        Produces plots of the dendrogram from hierarchical clustering, and overview
        plots of the clustering results.
        
        Inputs:
        ------------------------------------------------------------------------
        cells    list
                 the list of cells whose STAs should go into the clustering analysis
        n_clusters    number of clusters that should be retained; one should
                      experiment with this after looking at the data
        marker_code    dictionary    
                       if the dictionary with cell IDs as keys and their corr
                       esponding bias indexes as values is passed, then the markers
                       in the overview plot will reflect bias index of the cell
        
        n_components    float or int
                        number of components to keep in pca; for more detail see
                        documentation of sklearn.decomposition.PCA
        path    string
                specifying the path where to save the figures; if None, don't
                save the figures
        
        Outputs:
        -----------------------------------------------------------------------
        None;
        sets
        self.colors_dict    dictionary
                            with cell IDs as keys and color references as values
        self.color_mapping    dictionary that matches color references from 
                            colors_dict to actual matplotlib.pyplot colors 
        self.n_clusters
        self.cluster_labels    dictionary
                                with cell IDs as keys and cluster number as values
                                
        '''
#         X_input = [np.transpose(value) for value in self.sta.values()]
        X_input = [np.transpose(self.sta[cell]) for cell in cells]
        
        X_input = np.average(np.asarray(X_input), axis = 2)
        if znorm_this:
            for i, cell in enumerate(cells):
                X_input[i, :] = self.znorm(X_input[i, :])
        self.n_clusters = n_clusters

        new_cells = deepcopy(cells)
        boolean_mask = np.ones(X_input.shape[0], dtype=bool)
        for cell_no, cell_id in enumerate(cells):
            if np.any(np.isnan(X_input[cell_no])):
                new_cells.remove(cell_id)
                boolean_mask[cell_no] = False
                print(cell_id)
        
        X_input = X_input[boolean_mask, :]

        my_pca = PCA(n_components=n_components, svd_solver='full')
        my_pca.fit(X_input)
        temp = np.transpose([my_pca.components_[pcs[0]], \
                                 my_pca.components_[pcs[1]]])
            
        X_output = np.dot(X_input-np.mean(X_input, axis=0), temp)

        
        if pkl_path is not None:
            if mode == 'write':
                pca_and_linkage = {'pca' : my_pca}
                with open(pkl_path+'pca_and_linkage_matrix.pkl', 'wb') as f:
                    pkl.dump(pca_and_linkage, f)
            elif mode == 'read': 
                with open(pkl_path+'pca_and_linkage_matrix.pkl', 'rb') as f:
                    pca_and_linkage = pkl.load(f)
                    my_pca = pca_and_linkage['pca']
                    X_output = my_pca.transform(X_input)
        
        print('Number of components : {}'.format(my_pca.n_components_))
        print('explained variance ratio : {}'.format(my_pca.explained_variance_ratio_))
         
        if use_pca_output:
            basis = np.zeros((X_input.shape[1], len(pcs)))
            for i, pc in enumerate(pcs):
                if pc == 1:
                    basis[:, i] = -np.transpose(my_pca.components_[pc])
                else:
                    basis[:, i] = np.transpose(my_pca.components_[pc])
            X_output = np.dot(X_input-np.mean(X_input, axis=0),basis)
#             X_output = np.dot(X_input, np.transpose(my_pca.components_[0::2]))
            cluster_input = X_output
        else:
            cluster_input = X_input
        cluster_method = method
        cluster_metric = metric
        Z = linkage(cluster_input, method = cluster_method, metric = cluster_metric)
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        labels = labels-1
        for i, cell in enumerate(new_cells):
            self.cluster_labels[cell] = labels[i]
        #pyplot version
        ######################################################################
        dark_green_hex = sns.color_palette('PiYG', n_colors=10).as_hex()[-1]
        dark_red_hex =  sns.color_palette('YlOrRd', n_colors=10).as_hex()[-1]
        dark_blue_hex = sns.color_palette('RdBu', n_colors=10).as_hex()[-1]
        dark_purple_hex =  sns.color_palette('PuOr', n_colors=10).as_hex()[-1]
        if alternative_color_scheme:
            dark_blue_hex = sns.color_palette('YlGnBu', n_colors=10).as_hex()[-5]
            dark_purple_hex = sns.color_palette('OrRd', n_colors=10).as_hex()[-5]
        choices_string = [np.repeat(dark_green_hex, labels.shape),  \
                          np.repeat(dark_red_hex, labels.shape),\
#                           np.repeat(dark_blue_hex, labels.shape), \
                          np.repeat(dark_purple_hex, labels.shape), \
                          np.repeat('k', labels.shape)]

        color_strings = [dark_green_hex, dark_red_hex, dark_purple_hex, \
                                dark_blue_hex, 'k']
        color_palette = [color_strings[c] for c in color_order]
        set_link_color_palette(color_palette)
        plt.style.use('seaborn-paper')   
#         plt.style.use('seaborn-dark-palette')
        ############do the actual clustering##############################
        
        fig = plt.figure(figsize=(16, 9 ))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('cell ID')
        ylabel = cluster_method+' '+cluster_metric+' distance'
        plt.ylabel(ylabel)
        color_threshold = Z[-n_clusters+1, 2]
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=12.,  # font size for the x axis labels
            labels = new_cells,
            above_threshold_color = 'k',
            distance_sort = True,
            color_threshold=color_threshold,
        )
        plt.gca().spines["top"].set_visible(False)  
        plt.gca().spines["left"].set_visible(False)  
        plt.gca().spines["right"].set_visible(False)  
        plt.gca().spines["bottom"].set_visible(False)
        fname = fname_prefix+'_lnp_dendrogram'
        if path is not None:
#             plt.savefig(path+fname+'.ps', dpi = 300, fmt='ps')
            plt.savefig(path+fname+'.png', dpi = 300, fmt='png')
            plt.savefig(path+fname+'.svg', dpi = 300, fmt='svg')
            plt.close(fig)
        dark_green = [sns.color_palette('PiYG', n_colors=10)[-1]]
        dark_red = [sns.color_palette('YlOrRd', n_colors=10)[-1]]
        dark_blue = [sns.color_palette('RdBu', n_colors=10)[-1]]
        dark_purple =  [sns.color_palette('PuOr', n_colors=10)[-1]]
        if alternative_color_scheme:
            dark_blue = [sns.color_palette('YlGnBu', n_colors=10)[-5]]
            dark_purple = [sns.color_palette('OrRd', n_colors=10)[-5]]
        color_mapping = dict(zip(['dark_green', 'dark_red', 'dark_purple', \
                                  'dark_blue','k'],
                               [dark_green, dark_red, dark_purple, dark_blue, 'k']))
        
        choices = [np.repeat('dark_green', labels.shape), 
                    np.repeat('dark_red', labels.shape),\
                   np.repeat('dark_purple', labels.shape),\
                   np.repeat('dark_blue', labels.shape),
                np.repeat('k', labels.shape)]
        choices = [choices[c] for c in color_order]
        if marker_code is None:
            marker_choices = [np.repeat('o', labels.shape),  np.repeat('o', labels.shape),\
                   np.repeat('o', labels.shape), np.repeat('o', labels.shape),
                              np.repeat('o', labels.shape)]
            colors = np.choose(labels, choices)
            markers = np.choose(labels, marker_choices)
            
        else:
            marker_choices = [np.repeat('d', labels.shape),
                              np.repeat('o', labels.shape),
                              np.repeat('x', labels.shape),
                              np.repeat('s', labels.shape)]
            
            a = [0 if marker_code[cell] == -2
                 else 1 if marker_code[cell]<-0.25
                 else 3 if marker_code[cell]>0.5
                 else 2
                 for cell in new_cells]
            markers = np.choose(a, marker_choices)
            colors = np.choose(labels, choices)

        for i, cell in enumerate(new_cells):
            self.colors_dict[cell] = colors[i]

        self.color_mapping = color_mapping
        x = np.linspace(-self.delta_t_to_spike_back, \
                        -self.delta_t_to_spike_forward, self.len_sta_samples)
        
        
        '''
        #plot 1st vs 2nd component
        '''
        fig = plt.figure()

        ax = plt.subplot2grid([3,4], [0,1], rowspan=3, colspan=3)
        for i in range(X_output.shape[0]):
            cl = color_mapping[colors[i]]
            
            
#             alpha = \
#             max([np.average(spikes.benchmark[new_cells[i]],axis = 0),0])
            alpha = 1
            ax.scatter(X_output[i, 0], X_output[i, 1], color = cl, \
                       marker = markers[i], alpha = alpha)
            try:
                if new_cells.index(annotate_cell) == i:
                    plt.arrow(X_output[i, 0]-0.1, X_output[i, 1]-0.1, 0.05, 0.05, width = 0.01, color = 'k')
#                 plt.show()
            except:
                pass
        
        if additional_STA_object is not None:
            pos_proj = []
            neg_proj = []
            add_cells = list(additional_STA_object.sta.keys())
            pca_input = [np.transpose(additional_STA_object.sta[cell]) for\
                          cell in add_cells]
        
            pca_input = np.average(np.asarray(pca_input), axis = 2)
            

            new_add_cells = deepcopy(add_cells)
            boolean_mask = np.ones(pca_input.shape[0], dtype=bool)
            for cell_no, cell_id in enumerate(add_cells):
                if np.any(np.isnan(pca_input[cell_no])):
                    new_add_cells.remove(cell_id)
                    boolean_mask[cell_no] = False
            
            pca_input = pca_input[boolean_mask, :] 
#             pca_output = my_pca.transform(pca_input)
            pca_output = np.dot(pca_input-np.mean(pca_input, axis=0),\
                                np.transpose([my_pca.components_[pcs[0]],\
                                              -my_pca.components_[pcs[1]]]))
            for i in range(pca_output.shape[0]):
                ax.scatter(pca_output[i, 0], pca_output[i, 1], color = 'k',\
                           alpha = 0.5, marker = '.')
                
        '''
        plot the PC projections of all cells whose STA was not used for PC and
        clustering
        '''
#         left_out_cells  = [cell for cell in self.sta.keys() if not(cell in cells)]
# 
#         pca_input = [np.transpose(self.sta[cell]) for\
#                       cell in left_out_cells]
#     
#         pca_input = np.average(np.asarray(pca_input), axis = 2)
#         
# 
#         new_left_out_cells = deepcopy(left_out_cells)
#         boolean_mask = np.ones(pca_input.shape[0], dtype=bool)
#         for cell_no, cell_id in enumerate(left_out_cells):
#             if np.any(np.isnan(pca_input[cell_no])):
#                 new_left_out_cells.remove(cell_id)
#                 boolean_mask[cell_no] = False
#         
#         pca_input = pca_input[boolean_mask, :] 
#         pca_output = my_pca.transform(pca_input)
#         for i in range(pca_output.shape[0]):
#             alpha = max([np.average(spikes.benchmark[new_left_out_cells[i]],axis = 0),0])
#             
#             ax.scatter(pca_output[i, 0], pca_output[i, 1], facecolor = 'k', \
#                        edgecolor = '0.5', alpha = 0.2)
#             ax.scatter(pca_output[i, 0], pca_output[i, 1], color = 'k',\
#                        alpha = 0.2)



        ax.yaxis.tick_right()
        for label in ax.get_xticklabels()[::2]:
            label.set_visible(False)
        for label in ax.get_yticklabels()[::2]:
            label.set_visible(False)
        ax.spines["top"].set_visible(False)  
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
 
        ax.spines["right"].set_visible(False)  
        
        ax.tick_params(axis = 'both', which ='both', bottom = False, left = False, right=False, top = False)
#         ax.yaxis.labelpad = -75
        ax.set_xlabel('projection on PC ' + str(pcs[0]+1))
        ax.set_ylabel('projection on PC ' + str(pcs[1]+1))
        
        ax.yaxis.set_label_position("right")
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        transform_obj = ax.transData
        horizontal_scalebar = AnchoredSizeBar(transform_obj, 1,\
                           '1 AU', 'upper right', 
                           pad=1,
                           frameon=False
                           )
        
                        

        
        '''
        ############### plot the cluster STAs ##########################
        
        '''
        ylims = (-0.2, 0.2)
        n_subplots = self.n_clusters+len(pcs)+1
        axes_sta = []
        for cluster in range(n_clusters):
            new_ax = fig.add_subplot(n_subplots,4,4*(cluster)+1)
            axes_sta.append(new_ax)
        sta_all = {cluster:[] for cluster in range(self.n_clusters)}
        for i, cell in enumerate(new_cells):

            ax = axes_sta[labels[i]]
            plt.sca(ax)
            sta_white = np.average(self.sta[cell], 0)
            cl = color_mapping[colors[i]][0]
            plt.plot(x, sta_white, alpha = 0.3, color = cl, linewidth = 0.5)
            
            sta_all[labels[i]].append(sta_white)
        cluster_colors = list(color_mapping.values())
        cluster_colors = [cluster_colors[c] for c in color_order]
        for cluster in range(self.n_clusters):
            
            sta_average = np.average(sta_all[cluster], axis =0)
            plt.sca(axes_sta[cluster])
            plt.plot(x, sta_average, color = cluster_colors[cluster][0])
#             if cluster >0:
#                 plt.ylim(axes_sta[cluster-1].get_ylim())
            ax = axes_sta[cluster]
            for label in ax.get_xticklabels():
                label.set_visible(False)
            for label in ax.get_yticklabels():
                label.set_visible(False)    
                
            ax.tick_params(axis = 'both', which ='both', bottom = False, left = False, right=False, top = False)
            ax.spines["top"].set_visible(False)    
            ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(False) 
            ax.spines["bottom"].set_visible(False)
            ax.axhline(y=0, ls = '--', color='k')
            ax.axvline(x=0, color='k') 
#             ax.set_ylim(ylims)
#         axes_sta[0].set_ylim(axes_sta[1].get_ylim())
        transform_obj = ax.transData
        horizontal_scalebar = AnchoredSizeBar(transform_obj, 5,\
                            label='5 ms', 
#                             label='',
                           loc = 'lower left', 
                           borderpad=-1, label_top=True, 
                           frameon=False, size_vertical=0.02
                           )
        ax.add_artist(horizontal_scalebar)
            
        for ax in axes_sta:
            transform_obj = ax.transData
#             horizontal_scalebar = AnchoredSizeBar(transform_obj, 2,\
#                                '2 ms', 'lower left', 
#                                borderpad=-1, label_top=True, 
#                                frameon=False, size_vertical=0.005
#                                )
            
            vertical_scalebar = AnchoredSizeBar(transform_obj, 0.4,\
#                                r'0.1 $\frac{mA}{cm^2}$', 
                                loc = 'upper left', label = '',
                               frameon=False,
                               label_top=True, borderpad=-1,
                               size_vertical=0.1, pad = 0.2
                               )
#             ax.add_artist(horizontal_scalebar)
#             ax.add_artist(vertical_scalebar)
        
        '''
        #pca 1
        #########################################################################
        '''
        ax = fig.add_subplot(n_subplots,4,4*(self.n_clusters)+1)
        plt.plot(x, my_pca.components_[pcs[0]], color = 'k')
        for label in ax.get_xticklabels():
            label.set_visible(False)
        for label in ax.get_yticklabels():
            label.set_visible(False)    
            
        ax.tick_params(axis = 'both', which ='both', bottom = False, left = False, right=False, top = False)
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False) 
        ax.spines["bottom"].set_visible(False)
        ax.axhline(y=0, ls = '--', color='k')
        ax.axvline(x=0, color='k') 
#         plt.annotate('PC 1', (0.1, 0.4), xycoords = 'axes fraction')
        '''
        #pca 2
        #########################################################################
        '''
        ax = fig.add_subplot(n_subplots,4,4*(self.n_clusters+1)+1)
        plt.plot(x, -my_pca.components_[pcs[1]], color = 'k')
        for label in ax.get_xticklabels():
            label.set_visible(False)
        for label in ax.get_yticklabels():
            label.set_visible(False)    
            
        ax.tick_params(axis = 'both', which ='both', bottom = False, left = False, right=False, top = False)
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False) 
        ax.spines["bottom"].set_visible(False)
    
                
        ax.axhline(y=0, ls = '--', color='k')
        ax.axvline(x=0, color='k')

        '''
        #pca 3
        #########################################################################
        '''
        ax = fig.add_subplot(n_subplots, 4, 4 * (self.n_clusters +2) + 1)
        plt.plot(x, my_pca.components_[pcs[2]], color='k')
        for label in ax.get_xticklabels():
            label.set_visible(False)
        for label in ax.get_yticklabels():
            label.set_visible(False)

        ax.tick_params(axis='both', which='both', bottom=False, left=False, right=False, top=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.axhline(y=0, ls='--', color='k')
        ax.axvline(x=0, color='k')
    #         plt.annotate('PC 2', (0.1, 0.4), xycoords = 'axes fraction')
#         path = r'D:\Lara analysis\2017-11-03 - bl6\plots\\'
    
        '''
        #add dendrogram
        #######################################################################
        '''
        fig.add_subplot(n_subplots,4,4*(self.n_clusters+3)+1)
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=12.,  # font size for the x axis labels
#             labels = new_cells,
            no_labels = True, 
            above_threshold_color = 'k',
            distance_sort = True,
            color_threshold=color_threshold,
        )
#         plt.gca().spines["top"].set_visible(False)  
#         plt.gca().spines["left"].set_visible(False)  
#         plt.gca().spines["right"].set_visible(False)  
#         plt.gca().spines["bottom"].set_visible(False) 
        plt.gca().set_axis_off()
        fname =fname_prefix+ '_LNP_cluster'
        if path is not None:
        
#             plt.savefig(path+fname+'.eps', dpi = 300, fmt='eps')
            plt.savefig(path+fname+'.png', dpi = 300, fmt='png')
            plt.savefig(path+fname+'.svg', dpi = 300, fmt='svg')
            plt.savefig(path+fname+'.pdf', dpi = 300, fmt='pdf')
            plt.close(fig)
        else:
#             plt.show()
            pass
#         plt.close(fig)

    def compute_peak_latency(self, cells, clustered=False, plot= False, plot_path = None):
        pos_peak_latency = {cell: 1 for cell in cells}
        neg_peak_latency = {cell: 1 for cell in cells}
        if clustered:
            for cell in cells:
                if (self.cluster_labels[cell] == 0):
                    temp = np.argmin(np.average(self.sta[cell], axis=0))
                    temp /= 10
                    temp = self.delta_t_to_spike_back - temp
                    neg_peak_latency[cell] = temp
                elif (self.cluster_labels[cell] == 1):
                    temp = np.argmax(np.average(self.sta[cell], axis=0)[:200])
                    temp /= 10
                    temp = self.delta_t_to_spike_back - temp
                    pos_peak_latency[cell] = temp
                else:
                    temp = np.argmin(np.average(self.sta[cell], axis=0))
                    temp /= 10
                    temp = self.delta_t_to_spike_back - temp
                    neg_peak_latency[cell] = temp
                    temp = np.argmax(np.average(self.sta[cell], axis=0)[:200])
                    temp /= 10
                    temp = self.delta_t_to_spike_back - temp
                    pos_peak_latency[cell] = temp
                
            self.pos_peak_latency = pos_peak_latency
            self.neg_peak_latency = neg_peak_latency

            d = {'cluster label' : [self.cluster_labels[cell] for cell in cells],
                 'peak latency' : [self.neg_peak_latency[cell] if \
                                   self.cluster_labels[cell] == 0 else\
                                   self.pos_peak_latency[cell]
                                   for cell in cells]}
            df = pd.DataFrame(data = d)

            for i in range(self.n_clusters):
    #             print('Cluster {} mean peak latency : {}, std: {}, median :  {}'.format(i,\
                print('Cluster {} median :  {}, min:{}, max:{}, std:{}'.format(i,\
    #                             df['peak latency'].loc[df['cluster label']==i].mean(),\
    #                             df['peak latency'].loc[df['cluster label']==i].std(),\
                                df['peak latency'].loc[df['cluster label']==i].median(),
                                df['peak latency'].loc[df['cluster label']==i].min(),
                                df['peak latency'].loc[df['cluster label']==i].max(),
                                df['peak latency'].loc[df['cluster label']==i].std()))

            mean_neg = np.average([self.neg_peak_latency[cell]  \
                                   for cell in cells if \
                                   self.cluster_labels[cell] == 2])
            std_neg = np.std([self.neg_peak_latency[cell]  \
                                   for cell in cells if \
                                   self.cluster_labels[cell] == 2])
            median_neg = np.median([self.neg_peak_latency[cell]  \
                                   for cell in cells if \
                                   self.cluster_labels[cell] == 2])
            min_neg = np.min([self.neg_peak_latency[cell]  \
                                   for cell in cells if \
                                   self.cluster_labels[cell] == 2])
            max_neg = np.max([self.neg_peak_latency[cell]  \
                                   for cell in cells if \
                                   self.cluster_labels[cell] == 2])


            print('Cluster 2  median: {}, min:{}, max:{}'.format( median_neg,\
                                                                  min_neg, max_neg))

        else:
            for cell in cells:
                beta = np.average(self.sta[cell], axis=0)
                temp = np.argmin(beta)
                temp /= 10
                temp = self.delta_t_to_spike_back - temp
                neg_peak_latency[cell] = temp
            self.neg_peak_latency = neg_peak_latency
            print('median: {}, min:{}, max:{}'.format(np.median(list(self.neg_peak_latency.values())),
                                                      np.min(list(self.neg_peak_latency.values())),
                                                      np.max(list(self.neg_peak_latency.values()))))

        if plot:
            
            dark_green = sns.color_palette('YlGn', n_colors=10)[-1]
            dark_red = sns.color_palette('OrRd', n_colors=10)[-1]
            dark_blue = sns.color_palette('RdBu', n_colors=10)[-1]
            my_pal = {0 : dark_green, 1:dark_red, 2:dark_blue}
            sns.swarmplot(x = df['cluster label'], y = df['peak latency'], \
                           palette = my_pal)
#             sns.violinplot(x = 'cluster label', y = 'peak latency', hue = 'hue',\
#                             data = df, split = True, palette = my_pal, \
#                             inner = 'points')
#             ax = sns.violinplot(x=df['cluster label'], y=df['peak latency'],\
#                                 inner='point', split=True,\
#                                 palette=my_pal)
            
            
            
            
#             
            fname='latency'
            if plot_path is not None:
                plt.savefig(plot_path+fname+'.png', dpi = 300, fmt='png')
    #             plt.savefig(path+fname+'.svg', dpi = 300, fmt='svg')
                plt.savefig(plot_path+fname+'.pdf', dpi = 300, fmt='pdf')

#             plt.show()

    def znorm(self, y_data):
        """
        Z-transform data
        """

        transformed_array = (y_data - y_data.mean()) / y_data.std()
        return transformed_array


        