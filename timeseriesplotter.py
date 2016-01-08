# timeseriesplotter.py
# Anders Berliner
# 20160107
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.widgets as widgets
from matplotlib.widgets import Slider
from itertools import izip
from sklearn.cross_validation import StratifiedShuffleSplit
from random import shuffle


class SpotTimePlot(object):
    # may add kargs for other plotting options
    def __init__(self, y, column_headers,
                rows_per_page=3, cols_per_page=4,
                subset=False, subset_size=10, n_trial_groups=3,
                verbose=False):

        self.y = y
        self.groups = y['classification'].unique()

        self.column_headers = column_headers
        self.column_hash = {v:i for i,v in enumerate(self.column_headers)}
        self.subset = subset
        self.subset_size = subset_size
        self.verbose = verbose

        # view-grouping options
        self.n_trial_groups = n_trial_groups
        self.rows_per_page = rows_per_page
        self.cols_per_page = cols_per_page

        # set-up the plot window
        self.fig, self.ax = plt.subplots(figsize=(12,8))
        plt.subplots_adjust(bottom=0.25)

        axcolor = 'lightgoldenrodyellow'
        self.axtrial = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)
        self.axspot = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        self.axclass = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

        # need to update to find the number of classes, trial_groups
        self.strial = Slider(self.axtrial, 'Trial', 0, n_trial_groups, valinit=0)
        self.sspot = Slider(self.axspot, 'Spot', 1, len(self.column_headers), valinit=1)
        self.sclass = Slider(self.axclass, 'Class', 0, len(self.groups), valinit=0)

        self.strial.on_changed(self.update)
        self.sspot.on_changed(self.update)
        self.sclass.on_changed(self.update)

        #plt.tight_layout()

    def update(self, val):
        spot = int(self.sspot.val)
        group = int(self.sclass.val)
        trial = int(self.strial.val)
        self.plot_fit(spot=spot, group=group, trial=trial)

    def get_column_index(self, column_name):
        # may not be necessary
        return self.column_hash[column_name]

    def get_column_name(self, column_index):
        return self.column_headers[column_index]


    def split_trial_groups(self):
        # split-up the data for the purposes of plotting into smaller
        # groups, assuring that each has elements from each class
        y = self.y
        sss = StratifiedShuffleSplit(y=y['classification'],
                                    n_iter=self.n_trial_groups,
                                    test_size=1.0/self.n_trial_groups,
                                    random_state=1)

        trial_groups = []
        for train_index, test_index in sss:
            trial_groups.append(test_index)

        return trial_groups

    def plot_fits(self, X, X_pred):
        import matplotlib
        self.colors = ['r', 'g', 'b']
        more_colors = matplotlib.colors.cnames.keys()
        shuffle(more_colors)
        # print self.colors
        # print more_colors
        self.colors.extend(more_colors)

        self.trial_groups = self.split_trial_groups()
        self.X = X
        self.X_pred = X_pred

        self.plot_fit(spot=1, group=0, trial=0)
        plt.show()

    def plot_fit(self, spot=1, group=0, trial=0):
        print 'S:%d, G:%d, T:%d' % (spot, group, trial)
        # raw, pred data, labels for this group of trials => trial
        raw_trials = self.X.iloc[self.trial_groups[trial]]
        pred_trials = self.X_pred.iloc[self.trial_groups[trial]]
        labels = self.y.iloc[self.trial_groups[trial]]

        # get only those with right label => group
        mask = labels['classification'] == self.groups[group]
        raw_trials = raw_trials[mask]
        pred_trials = pred_trials[mask]
        plt.sca(self.ax)
        plt.cla()
        # plot only this spot => spot
        for i, (raw, pred) in enumerate(izip(raw_trials, pred_trials)):
            # print i, raw
            t = raw[:,0] # 0th column is time
            raw = raw[:,spot]
            pred = pred[:,spot]

            # print i, t.shape, raw.shape, pred.shape
            plt.plot(t, raw, color=self.colors[i], marker='o', linestyle='', alpha=0.5)
            plt.plot(t, pred, color=self.colors[i])
            plt.title(('S:%s, G:%s, T:%s' % (self.get_column_name(spot),self.groups[group],trial)))
    def plot_raw(self):
        pass
