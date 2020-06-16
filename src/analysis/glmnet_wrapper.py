
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster, \
     set_link_color_palette
from scipy.spatial.distance import pdist
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd


class GlmCVWrapper():
    """
    Performs cross validation for finding hyperparamters for Logistic Regression 
    models.
    """
    def __init__(self,
                 lambdas = None,
                 penalty = 'elasticnet',
                 solver = 'saga',
                 multi_class = 'ovr',
                 l1_ratios = [0, 0.01, 0.1, 0.5, 1],
                 cv=3,
                 scoring='neg_log_loss',
                 n_jobs=1,
                 max_iter=100
                 ):
        """
        Sets the parameters for cross validation
        """
        if lambdas is None:
            lambdas = np.geomspace(10**(-4), 10**(-2), num = 5)
        self.lambdas = lambdas
        self.penalty = penalty
        self.solver = solver
        self.multi_class = multi_class
        self.l1_ratios = l1_ratios
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.max_iter = max_iter


    def cv_and_fit(self, stim, spikes, cells,
                   filter_length_samples=300, binsize=1,
                   pre_spike_time=20, post_spike_time=-10):

        '''
        perform cross validation over specified grid of hyperparameters and
        stores best performing parameters and model
        :param stim:
        :param spikes:
        :param cells:
        :param filter_length_samples:
        :param binsize:
        :param pre_spike_time:
        :param post_spike_time:
        :return:
        '''

        X = np.transpose(
            stim.create_stimulus_ensemble(
                filter_length_samples, binsize
            )
        )

        X = np.tile(X, (spikes.n_stim_reps, 1))
        y_all = spikes.create_spike_histogram(pre_spike_time, post_spike_time, \
                                      binsize = binsize)
        models = {cell: [] for cell in cells}
        reg_lambda = {cell: None for cell in cells}
        alpha = {cell: None for cell in cells}
        for cell in cells:
            y = np.concatenate(y_all[cell])
            mod = LogisticRegressionCV(Cs=self.lambdas,
                                       penalty=self.penalty,
                                       solver=self.solver,
                                       multi_class=self.multi_class,
                                       l1_ratios=self.l1_ratios,
                                       cv=self.cv,
                                       scoring=self.scoring,
                                       n_jobs=self.n_jobs,
                                       max_iter=self.max_iter
                                       )
            print(mod.n_jobs)
            mod.fit(X, y)
            models[cell] = mod
            reg_lambda[cell] = mod.C_[0]
            alpha[cell] = mod.l1_ratio_[0]
        self.models = models
        self.reg_lambda = reg_lambda
        self.alpha = alpha

class GlmWrapper():

    """
    Class to fit Logistic Regression model to cell responses
    """

    def __init__(self, penalty = 'elasticnet',
                 solver = 'saga',
                 multi_class = 'ovr'):
        self.penalty = penalty
        self.solver = solver
        self.multi_class = multi_class

    def fit_train_test(self,
                       glm_cv_wrapper,
                       stim,
                       spikes,
                       cells,
                       filter_length_samples=300,
                       binsize=1,
                       pre_spike_time=20,
                       post_spike_time=-10,
                       n_partition = 5):
        '''
        Fits a Logistic Regression model for the specified cells and saves the
        model, linear and nonlinear prediction and model performance as attributes
        of the class instance.

        :param glm_cv_wrapper: glm_cv_wrapper object
        :param stim: stimulus object
        :param spikes: spikes object
        :param cells: list of cells for which to fit model
        :param filter_length_samples: int; desired length of filter in samples
        :param binsize: int; binsize in ms (prediction window)
        :param pre_spike_time: int; time window in ms to consider before spike
        :param post_spike_time: int; time window in ms to consider after spike
        :param n_partition: int; number of partitions into which the data is divided
        :return:  -
        '''
        self.n_partition = n_partition
        self.training_score = {cell:[None for _ in range(len(spikes.partition))]
                               for cell in cells}
        self.models = {cell:[None for _ in range(len(spikes.partition))]
                       for cell in cells}
        stimulus_snippets = stim.create_stimulus_ensemble(filter_length_samples,
                                                          binsize,
                                                          white=False)

        train_stim_snips = []
        for partition in spikes.partition:
            temp1 = stimulus_snippets[:, :partition[0]]

            start = partition[1]
            stop = spikes.stim_dur - pre_spike_time + post_spike_time
            temp2 = \
                stimulus_snippets[:, start:stop]
            train_stim_snips.append(np.concatenate([temp1, temp2], axis=1))
        self.train_stim_snips = [np.transpose(train_stim_snip) for \
                                 train_stim_snip in train_stim_snips]

        for i, partition in enumerate(spikes.partition):
            X = np.tile(self.train_stim_snips[i], (spikes.n_stim_reps, 1))
            Y_all = spikes.create_spike_histogram(lower_limit=pre_spike_time,
                                                  upper_limit=post_spike_time,
                                                  mode='train',
                                                  binsize=binsize,
                                                  set_no=i)
            for cell in cells:
                Y = np.concatenate(Y_all[cell])
                m = LogisticRegression(penalty=self.penalty,
                                       C=glm_cv_wrapper.reg_lambda[cell],
                                       solver=self.solver,
                                       l1_ratio=glm_cv_wrapper.alpha[cell])
                m.fit(X, Y)
                self.models[cell][i] = m

        test_stim_snips = \
            [stimulus_snippets[:, p[0]:p[1]] for p in spikes.partition]
        self.lin_prediction = {cell: [np.zeros(test_stim_snip.shape[-1])
                                    for test_stim_snip in test_stim_snips]
                               for cell in cells}
        self.nonlin_prediction = {cell: [np.zeros(test_stim_snip.shape[-1])
                                for test_stim_snip in test_stim_snips]
                                  for cell in cells}
        self.lin_corr = {cell: [None for i in range(n_partition)]
                         for cell in cells}
        self.nonlin_corr = {cell: [None for i in range(n_partition)]
                         for cell in cells}

        for i, test_stim_snip in enumerate(test_stim_snips):
            Y_test_all = \
                spikes.create_spike_histogram(lower_limit=pre_spike_time,
                                              upper_limit=-post_spike_time,
                                              mode='test', set_no=i)
            X_test = test_stim_snip
            for cell in cells:
                m = self.models[cell][i]
                Y_test = np.average(Y_test_all[cell], axis=0)
                temp = m.predict_proba(np.transpose(X_test))
                temp = temp[:, 1]
                self.nonlin_prediction[cell][i] = temp
                temp = np.matmul(np.transpose(X_test), np.squeeze(m.coef_)) + m.intercept_[0]
                self.lin_prediction[cell][i] = temp
                self.lin_corr[cell][i] = np.round(np.corrcoef(temp,
                                                              Y_test,
                                                              )[0][1], 2)
                nonlin_temp = self.nonlin_prediction[cell][i]
                self.nonlin_corr[cell][i] = \
                    np.round(np.corrcoef(nonlin_temp, Y_test,
                                         )[0][1], 2)



    def hierarchical_cluster(self, cells, n_clusters, pcs=[0, 1],
                             pca=True, cluster_method='average',
                             cluster_metric='euclidean', n_components=0.9,
                             znorm_this=False):
        '''
        Perform hierarchical clustering of cells based on (the projection of)
        their filters found by fitting the Logistic Regression model.

        :param cells: list of cells;
        :param n_clusters: int; number of desired clusters
        :param pcs: list of ints; which principal components to use for projecting the filters
        :param pca: bool; whether to perform PCA and do clustering on projection
        :param cluster_method: string; specification of clustering method to use
                (passed on to scipy.cluster.hierarchy.linkage)
        :param cluster_metric: string; specification of clustering metric to use
                (passed on to scipy.cluster.hierarchy.linkage)
        :param n_components: int or float; how many components to keep in PCA
                (passed on to sklearn.decomposition.PCA
        :param znorm_this: whether to znorm the filters before PCA and clustering
        :return:
        '''

        self.n_clusters = n_clusters
        X_input = []
        cluster_labels = {cell: -1 for cell in cells}
        new_cells = deepcopy(cells)
        for cell in cells:
            all_betas = [self.models[cell][i].coef_ for i in
                         range(self.n_partition)]
            beta = np.squeeze(np.average(all_betas, axis=0))
            if znorm_this:
                beta = self.znorm(beta)
            X_input.append(beta)
            # if np.sum(beta) != 0:
            #     X_input.append(beta)
            # else:
            #     new_cells.remove(cell)

        self.clustered_cells = new_cells
        X_input = np.squeeze(np.asarray(X_input))
        my_pca = PCA(n_components=n_components, svd_solver='full')
        my_pca.fit(X_input)
        if pca:
            basis = np.zeros((X_input.shape[1], len(pcs)))
            for i, pc in enumerate(pcs):
                basis[:, i] = np.transpose(my_pca.components_[pc])
            X_output = np.dot(X_input - np.mean(X_input, axis=0), basis)
            print('Number of components : {}'.format(my_pca.n_components_))
            print('explained variance ratio : {}'.format(my_pca.explained_variance_ratio_))
        else:
            X_output=X_input
        ############do the actual clustering##############################
        cluster_method = cluster_method
        cluster_metric = cluster_metric
        Z = linkage(X_output, method=cluster_method, metric=cluster_metric)
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        labels = labels - 1
        for i, cell in enumerate(new_cells):
            cluster_labels[cell] = labels[i]
        self.cluster_labels = cluster_labels
        self.cluster_method = cluster_method
        self.cluster_metric = cluster_metric
        self.n_clusters = n_clusters
        self.linkage_matrix = Z
        self.pca_input = X_input
        self.pca = my_pca
        self.pcs = pcs

    def clustering_plot(self, x_start, x_stop, marker_code=None, path=None,
                        annotate_cell=None):
        '''
        #plotting starts here
        '''
        assert hasattr(self, "cluster_labels"), "You haven't performed hierarchical clustering on this instance of GlmWrapper yet"
        light_green = [sns.color_palette('PiYG', n_colors=10)[-2]]
        light_red = [sns.color_palette('YlOrRd', n_colors=10)[-2]]
        light_blue = [sns.color_palette('RdBu', n_colors=10)[-2]]
        light_purple = [sns.color_palette('PuOr', n_colors=10)[-2]]
        color_mapping = \
            dict(zip(['light_green', 'light_red', 'light_purple', 'light_blue', 'k'], \
                     [light_green, light_red, light_purple, light_blue, 'k']))

        light_green_hex = sns.color_palette('PiYG', n_colors=10).as_hex()[-2]
        light_red_hex = sns.color_palette('YlOrRd', n_colors=10).as_hex()[-2]
        light_blue_hex = sns.color_palette('RdBu', n_colors=10).as_hex()[-2]
        light_purple_hex = sns.color_palette('PuOr', n_colors=10).as_hex()[-2]

        set_link_color_palette([light_green_hex, light_red_hex, \
                                light_purple_hex, light_blue_hex, 'k'])

        plt.style.use('seaborn-paper')
        fig = plt.figure(figsize=(16, 9))
        #     plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('cell ID')
        ylabel = self.cluster_method + ' ' + self.cluster_metric + ' distance in PCA space'
        plt.ylabel(ylabel)

        color_threshold = self.linkage_matrix[-self.n_clusters + 1, 2]

        dendrogram(
            self.linkage_matrix,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=15.,  # font size for the x axis labels
            labels=self.clustered_cells,
            above_threshold_color='k',
            distance_sort=True,
            color_threshold=color_threshold
        )
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        #     plt.show()
        #

        fname = 'GLM_dendrogram'
        if path is not None:
            plt.savefig(path + fname + '.png', dpi=300, fmt='png')
            plt.close(fig)
        else:
            plt.show()

        choices = [np.repeat('light_green', len(self.clustered_cells)),
                   np.repeat('light_red', len(self.clustered_cells)),
                   np.repeat('light_purple', len(self.clustered_cells)),
                   np.repeat('light_blue', len(self.clustered_cells)),
                   np.repeat('k', len(self.clustered_cells))]

        cluster_label_list = [self.cluster_labels[cell] for cell in self.clustered_cells]
        if marker_code is None:
            marker_choices = [np.repeat('o', len(self.clustered_cells)),
                              np.repeat('o', len(self.clustered_cells)),
                              np.repeat('o', len(self.clustered_cells)),
                              np.repeat('o', len(self.clustered_cells))]
            colors = np.choose(cluster_label_list, choices)
            markers = np.choose(cluster_label_list, marker_choices)

        else:
            marker_choices = [np.repeat('x', len(self.clustered_cells)),
                              np.repeat('d', len(self.clustered_cells)),
                              np.repeat('o', len(self.clustered_cells)),
                              np.repeat('*', len(self.clustered_cells))]
            markers = np.choose(marker_code, marker_choices)
            colors = np.choose(cluster_label_list, choices)

        colors_dict = {cell: 'k' for cell in self.clustered_cells}
        for i, cell in enumerate(self.clustered_cells):
            colors_dict[cell] = colors[i]
        self.colors_dict = colors_dict
        self.color_mapping = color_mapping
        markers_dict = dict(zip(self.clustered_cells, list(markers)))
        x = np.linspace(x_start, x_stop, self.pca_input.shape[1])

        '''
        plot 1st vs 2nd component
        '''
        fig = plt.figure()
        ######## plot 2nd principal component

        temp = np.transpose([self.pca.components_[self.pcs[0]],
                             self.pca.components_[self.pcs[1]]])

        X_output = np.dot(self.pca_input - np.mean(self.pca_input, axis=0), temp)

        ax = plt.subplot2grid([3, 4], [0, 1], rowspan=3, colspan=3)
        for i, cell in enumerate(self.clustered_cells):
            cl = color_mapping[colors_dict[cell]]
            m = markers_dict[cell]
            ax.scatter(X_output[i, 0], X_output[i, 1], color=cl,
                       marker=m, label='a')

            try:
                if self.clustered_cells.index(annotate_cell) == i:
                    plt.arrow(X_output[i, 0] - 0.02, X_output[i, 1] - 0.02, 0.01, 0.01, \
                              width=0.002, color='k')
            except:
                pass
        ax.yaxis.tick_right()
        for label in ax.get_xticklabels()[::2]:
            label.set_visible(False)
        for label in ax.get_yticklabels()[::2]:
            label.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='both', which='both', bottom=False, left=False, right=False, top=False)

        ax.set_xlabel('projection on PC {}'.format(self.pcs[0]+1))
        ax.set_ylabel('projection on PC {}'.format(self.pcs[1]+1))
        ax.yaxis.set_label_position("right")
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        transform_obj = ax.transData

        '''
        ############### plot the cluster STAs ##########################

        '''
        ylims = (-0.05, 0.05)
        n_subplots = self.n_clusters + len(self.pcs) + 1
        axes_sta = []
        for cluster in range(self.n_clusters):
            new_ax = fig.add_subplot(n_subplots, 4, 4 * (cluster) + 1)
            axes_sta.append(new_ax)
        sta_all = {cluster: [] for cluster in range(self.n_clusters)}
        for i, cell in enumerate(self.clustered_cells):
            ax = axes_sta[self.cluster_labels[cell]]
            plt.sca(ax)
            sta_white = self.pca_input[i, :]
            cl = color_mapping[colors_dict[cell]][0]
            plt.plot(x, sta_white, alpha=0.2, color=cl, linewidth=0.5)

            sta_all[self.cluster_labels[cell]].append(sta_white)
        cluster_colors = list(color_mapping.values())
        for cluster in range(self.n_clusters):
            if len(sta_all[cluster]) > 1:
                sta_average = np.average(sta_all[cluster], axis=0)
            else:
                sta_average = sta_all[cluster][0]
            plt.sca(axes_sta[cluster])
            plt.plot(x, sta_average, color=cluster_colors[cluster][0])
            ax = axes_sta[cluster]
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

        transform_obj = ax.transData
        horizontal_scalebar = AnchoredSizeBar(transform_obj, 5,
                                              '5 ms', 'lower left',
                                              borderpad=-1, label_top=True,
                                              frameon=False, size_vertical=0.005
                                              )
        ax.add_artist(horizontal_scalebar)
        '''
        #pca 1
        #########################################################################
        '''
        ax = fig.add_subplot(n_subplots, 4, 4 * (self.n_clusters) + 1)
        plt.plot(x, self.pca.components_[self.pcs[0]], color='k')
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
        xlims = plt.gca().get_xlim()
        '''
        #pca 2
        #########################################################################
        '''
        ax = fig.add_subplot(n_subplots, 4, 4 * (self.n_clusters + 1) + 1)
        plt.plot(x, self.pca.components_[self.pcs[1]], color='k')
        for label in ax.get_xticklabels():
            label.set_visible(False)
        for label in ax.get_yticklabels():
            label.set_visible(False)

        ax.tick_params(axis='both', which='both', bottom=False, left=False, right=False, top=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        plt.xlim(xlims)
        ax.axhline(y=0, ls='--', color='k')
        ax.axvline(x=0, color='k')
        #         plt.annotate('PC 2', (0.1, 0.4), xycoords = 'axes fraction', size='large', weight='bold')

        '''
        #pca 2
        #########################################################################
        '''
        ax = fig.add_subplot(n_subplots, 4, 4 * (self.n_clusters + 2) + 1)
        plt.plot(x, self.pca.components_[self.pcs[2]], color='k')
        for label in ax.get_xticklabels():
            label.set_visible(False)
        for label in ax.get_yticklabels():
            label.set_visible(False)

        ax.tick_params(axis='both', which='both', bottom=False, left=False, right=False, top=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        plt.xlim(xlims)
        ax.axhline(y=0, ls='--', color='k')
        ax.axvline(x=0, color='k')
        '''
        #add dendrogram
        #######################################################################
        '''

        fig.add_subplot(n_subplots, 4, 4 * (self.n_clusters + 3) + 1)
        dendrogram(
            self.linkage_matrix,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=12.,  # font size for the x axis labels
            no_labels=True,
            above_threshold_color='k',
            distance_sort=True,
            color_threshold=color_threshold,
        )
        plt.gca().set_axis_off()
        if path is not None:
            fname = 'GLM_cluster'
            #     plt.savefig(path+fname+'.eps', dpi = 100, fmt='eps', transparent=True, frameon=False)
            plt.savefig(path + fname + '.png', dpi=300, fmt='png', frameon=False)
            plt.savefig(path + fname + '.svg', babox_inches='tight', dpi=300, fmt='svg', frameon=False)
            plt.savefig(path + fname + '.pdf', dpi=300, fmt='pdf', frameon=False)
            plt.close(fig)
        else:
            plt.show()

    def znorm(self, y_data):
        """
        Z-transform data
        """

        transformed_array = (y_data - y_data.mean()) / y_data.std()
        return transformed_array

    def compute_peak_latency(self, cells, pre_spike_time):
        pos_peak_latency = {cell: 1 for cell in cells}
        neg_peak_latency = {cell: 1 for cell in cells}
        lower = 100
        upper = 250
        pre_spike_time = pre_spike_time-lower/10
        if hasattr(self, 'cluster_labels'):

            for cell in cells:
                all_betas = [self.models[cell][i].coef_ for i in
                             range(self.n_partition)]
                beta = np.squeeze(np.average(all_betas, axis=0))
                beta = beta[lower:upper]
                if (self.cluster_labels[cell] == 0):
                    temp = np.argmin(beta)
                    temp /= 10
                    temp = pre_spike_time - temp
                    neg_peak_latency[cell] = temp
                elif (self.cluster_labels[cell] == 1):
                    temp = np.argmax(beta)
                    temp /= 10
                    temp = pre_spike_time - temp
                    pos_peak_latency[cell] = temp
                else:
                    temp = np.argmin(beta[10:])
                    temp /= 10
                    temp = pre_spike_time - temp
                    neg_peak_latency[cell] = temp
                    temp = np.argmax(beta)
                    temp /= 10
                    temp = pre_spike_time - temp
                    pos_peak_latency[cell] = temp

            self.pos_peak_latency = pos_peak_latency
            self.neg_peak_latency = neg_peak_latency

            d = {'cluster label': [self.cluster_labels[cell] for cell in cells],
                 'peak latency': [self.neg_peak_latency[cell] if \
                                      self.cluster_labels[cell] == 0 else \
                                      self.pos_peak_latency[cell]
                                  for cell in cells]}
            df = pd.DataFrame(data=d)

            for i in range(self.n_clusters):
                #             print('Cluster {} mean peak latency : {}, std: {}, median: {}'.format(i,\
                print('Cluster {}, median: {}, min:{}, max:{}'.format(i, \
                                                                      #                             df['peak latency'].loc[df['cluster label']==i].mean(),\
                                                                      #                             df['peak latency'].loc[df['cluster label']==i].std(), \
                                                                      df['peak latency'].loc[
                                                                          df['cluster label'] == i].median(),
                                                                      df['peak latency'].loc[
                                                                          df['cluster label'] == i].min(),
                                                                      df['peak latency'].loc[
                                                                          df['cluster label'] == i].max()))

            mean_neg = np.average([self.neg_peak_latency[cell] \
                                   for cell in cells if \
                                   self.cluster_labels[cell] == 2])
            std_neg = np.std([self.neg_peak_latency[cell] \
                              for cell in cells if \
                              self.cluster_labels[cell] == 2])
            median_neg = np.median([self.neg_peak_latency[cell] \
                                    for cell in cells if \
                                    self.cluster_labels[cell] == 2])
            min_neg = np.min([self.neg_peak_latency[cell] \
                              for cell in cells if \
                              self.cluster_labels[cell] == 2])

            max_neg = np.max([self.neg_peak_latency[cell] \
                              for cell in cells if \
                              self.cluster_labels[cell] == 2])

            print('Cluster 2 median: {}, min: {}, max:{}'.format(median_neg, \
                                                                 min_neg, max_neg))

        else:
            for cell in cells:
                all_betas = [self.models[cell][i].coef_ for i in
                             range(self.n_partition)]
                beta = np.squeeze(np.average(all_betas, axis=0))
                beta = beta[lower:upper]
                temp = np.argmin(beta)
                temp /= 10
                temp = pre_spike_time - temp
                neg_peak_latency[cell] = temp
            self.neg_peak_latency = neg_peak_latency
            print('median: {}, min:{}, max:{}'.format(np.median(list(self.neg_peak_latency.values())),
                                                      np.min(list(self.neg_peak_latency.values())),
                                                      np.max(list(self.neg_peak_latency.values()))))

