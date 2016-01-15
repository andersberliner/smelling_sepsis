# timeseriesplotter.py
# Anders Berliner
# 20160107
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.widgets as widgets
from matplotlib.widgets import Slider, Button, RadioButtons
from itertools import izip
from sklearn.cross_validation import StratifiedShuffleSplit
from random import shuffle
import numpy as np

class SpotTimePlot(object):
    # may add kargs for other plotting options
    def __init__(self, y, column_headers,
                rows_per_page=3, cols_per_page=4,
                subset=False, subset_size=10, n_trial_groups=1,
                verbose=False, averages=True):
        self.averages = averages
        self.y = y
        print y.groupby('classification').count()
        self.groups = y['classification'].unique()
        self.groups.sort()
        self.column_headers = column_headers
        self.column_hash = {v:i for i,v in enumerate(self.column_headers)}
        self.subset = subset
        self.subset_size = subset_size
        self.verbose = verbose

        # view-grouping options
        self.n_trial_groups = n_trial_groups
        self.rows_per_page = rows_per_page
        self.cols_per_page = cols_per_page

        #plt.tight_layout()

    def update_fits(self, val):
        spot = int(self.sspot.val)
        group = int(self.sclass.val)
        trial = int(self.strial.val)
        self.plot_fit(spot=spot, group=group, trial=trial)

    def update_raws(self,val):
        spot = int(self.sspot.val)
        if self.radio.value_selected == 'averages':
            self.averages = True
        else:
            self.averages = False

        self.plot_raw(spot=spot, averages=self.averages)

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

        # self.trial_groups = self.split_trial_groups()
        self.X = X
        self.X_pred = X_pred

        # set-up the plot window
        self.fig, self.ax = plt.subplots(figsize=(12,8))
        plt.subplots_adjust(bottom=0.25)

        # make sliders
        axcolor = 'lightgoldenrodyellow'
        self.axtrial = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)
        self.axspot = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        self.axclass = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

        # need to update to find the number of classes, trial_groups
        self.strial = Slider(self.axtrial, 'Trial', 0, n_trial_groups, valinit=0)
        self.sspot = Slider(self.axspot, 'Spot', 1, len(self.column_headers), valinit=1)
        self.sclass = Slider(self.axclass, 'Class', 0, len(self.groups), valinit=0)

        self.strial.on_changed(self.update_fits)
        self.sspot.on_changed(self.update_fits)
        self.sclass.on_changed(self.update_fits)

        self.plot_fit(spot=1, group=0, trial=0, averages=self.averages)
        plt.show()

    def plot_fit(self, spot=1, group=0, trial=0):
        print 'S:%d, G:%d, T:%d' % (spot, group, trial)
        # raw, pred data, labels for this group of trials => trial
        # raw_trials = self.X.iloc[self.trial_groups[trial]]
        # pred_trials = self.X_pred.iloc[self.trial_groups[trial]]
        # labels = self.y.iloc[self.trial_groups[trial]]

        raw_trials = self.X
        pred_trials = self.X_pred
        labels = self.y

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

    def plot_raws(self, X, averages=True):
        import matplotlib
        self.colors = ['r', 'g', 'b']
        more_colors = matplotlib.colors.cnames.keys()
        shuffle(more_colors)
        # print self.colors
        # print more_colors
        self.colors.extend(more_colors)

        # set-up the plot window
        self.fig, self.ax = plt.subplots(figsize=(12,8))
        plt.subplots_adjust(bottom=0.15)

        # spot slider
        axcolor = 'lightgoldenrodyellow'
        self.axspot = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)
        self.sspot = Slider(self.axspot, 'Spot', 1, len(self.column_headers), valinit=1)
        self.sspot.on_changed(self.update_raws)

        # averages button
        # self.axavgbut = plt.axes([0.25, 0.025, 0.1, 0.04])
        # self.button = Button(self.axavgbut, 'Averages?', color=axcolor, hovercolor='0.975')
        # self.button.on_clicked(self.update_raws)

        self.rax = plt.axes([0.025, 0.05, 0.15, 0.15], axisbg=axcolor)
        self.radio = RadioButtons(self.rax, ('averages', 'all'), active=0)
        self.radio.on_clicked(self.update_raws)

        # self.trial_groups = self.split_trial_groups()
        self.X = X
        # self.X_pred = X_pred
        self.averages = averages
        self.plot_raw(spot=1, group=0, trial=0, averages=averages)
        plt.show()

    def one_spot(self, x, spot):
        t = x[:,0]
        raw = x[:,spot]
        # print raw.shape
        return raw
    def plot_raw(self, spot=1, group=0, trial=0, averages=True):
        raw_trials = self.X
        # print len(raw_trials), len(self.y)
        # print self.groups
        print self.averages, self.radio.value_selected
        # pred_trials = self.X_pred
        labels = self.y
        plt.sca(self.ax)
        plt.cla()

        t = self.X.iloc[0][:,0]
        for i, group in enumerate(self.groups):
            # print group
            # get only those with right label => group
            # print self.groups[group]
            # print self.groups
            # print labels['classification']
            mask = labels['classification'] == group
            grp_trials = raw_trials[mask]
            # pred_trials = pred_trials[mask]
            # print len(grp_trials)
            C = grp_trials.apply(lambda x: self.one_spot(x, spot))
            if averages:

                # print C.head()
                # print C
                AVG = np.array(C.iloc[0])
                # print AVG
                for i in xrange(1, len(C)):
                    AVG = np.vstack((AVG, C.iloc[i]))

                # print C.head()
                # AVG.shape
                toplot = np.mean(AVG, axis=0)
                # print toplot, t
                # print t.shape, toplot.shape
                if group == 'Control':
                    plt.plot(t, toplot, 'ko', label=group, linewidth='3')
                else:
                    plt.plot(t, toplot, label=group)
            else:
                for j, data in enumerate(C):
                    if group == 'Control':
                        if j==0:
                            plt.plot(t, data, color='black', linewidth='3', label=group)
                        else:
                            plt.plot(t, data, color='black', linewidth='3')
                    else:
                        if j==0:
                            plt.plot(t, data, color=self.colors[i], alpha=0.3, label=group)
                        else:
                            plt.plot(t, data, color=self.colors[i], alpha=0.3)

        plt.title(('S:%s, averages:%s' % (self.get_column_name(spot),self.averages)))
        # plt.ylim([-200,200])
        plt.legend(loc=0, fontsize=6)
