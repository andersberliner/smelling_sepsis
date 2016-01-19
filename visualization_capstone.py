import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from output_capstone import print_to_file_and_terminal as ptf
from itertools import izip

def make_series_plots(sm):
    pass

def find_cutoff_thresholds(df, cutoffs = [0.25, 0.5, 0.75]):
    probas = df['thresholds'].values[0]
    # probas are sorted from lowest to highest.  len(probabs > cutoff) is as
    # given, then zero reference the index
    i_25 = np.sum(probas > cutoffs[0]) - 1
    i_50 = np.sum(probas > cutoffs[1]) - 1
    i_75 = np.sum(probas > cutoffs[2]) - 1

    return i_25, i_50, i_75

def roc_plot(ax, sm, fold, t, testtrain = 'test', cutoffs = [0.25, 0.5, 0.75]):
    if testtrain == 'test':
        df = sm.trigger_scores_test
    elif testtrain == 'train':
        df = sm.trigger_scores
    df = df[(df['fold'] == fold) & (df['time'] == t)]
    x = df['fpr'].values[0]
    y = df['tpr'].values[0]

    # print x.shape, y.shape
    # auc = roc_auc_score()
    if fold == 'all':
        ax.plot(x,y, label=str(fold), linewidth=5, alpha=0.75)
        i_25, i_50, i_75 = find_cutoff_thresholds(df, cutoffs = [0.35, 0.5, 0.9])

        # add lines of various cut-offs
        ax.vlines(x[i_25], 0, 1, color='orchid', linestyle='--', linewidth=3, alpha=0.5, label=str(cutoffs[0]))
        ax.vlines(x[i_50], 0, 1, color='mediumseagreen', linestyle='--', linewidth=3, alpha=0.5, label=str(cutoffs[1]))
        ax.vlines(x[i_75], 0, 1, color='steelblue', linestyle='--', linewidth=3, alpha=0.5, label=str(cutoffs[2]))

        # annotate the plot
        text = ''
        for i, (label, col) in enumerate(izip(['F1', 'ACC', 'PREC', 'REC', 'AUC'],
            ['f1', 'overall_accuracy', 'precision', 'recall', 'auc'])):

            if i == 0:
                text = '%s %s' % (('%s:' % label).ljust(7), ('%0.3f' % df[col]).rjust(5))
            else:
                text += '\n%s %s' % (('%s:' % label).ljust(7), ('%0.3f' % df[col]).rjust(5))
        # text = 'F1:    %0.3f\nACC:   %0.3f\nPREC:  %0.3f\nREC:   %0.3f\nAUC:   %0.3f' % (df['f1'],
        #     df['overall_accuracy'],df['precision'], df['recall'], df['auc'])
        ax.text(0.20, 0.02, text, verticalalignment='bottom', horizontalalignment='left',
        fontsize=15)
    else:
        ax.plot(x,y, label=str(fold))

def roc_plot_setup(ax, t, runid, testtrain='test'):
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('%s - ROC - %s - %d' % (runid, testtrain, t))


def metric_vs_time(alldf, runid, title, debug=False):
    t = alldf['time'].values*20./60.
    plt.figure()
    plt.plot(t, alldf['f1'], label='f1', linewidth=3, alpha=0.5)
    plt.plot(t, alldf['recall'], label='recall', linewidth=3, alpha=0.5)
    plt.plot(t, alldf['precision'], label='precision', linewidth=3, alpha=0.5)
    plt.plot(t, alldf['auc'], label='auc', linewidth=3, alpha=0.5)
    plt.plot(t, alldf['accuracy'], label='accuracy', linewidth=3, alpha=0.5)
    plt.legend(loc=4)
    plt.xlabel('time (hrs)')
    plt.ylabel('Metric')
    plt.ylim([0.4,1])
    plt.title(runid + '-' + title)
    plt.savefig(runid + '/' + runid + '-' + title, dpi=200)
    if debug:
        plt.show()
    else:
        plt.close()

def make_trigger_plots(sm, y, runid, debug=False, logfile=None):
    # add trial to y for later merging
    y['trial'] = y.index


    # roc plots
    for t in sm.times:
        ptf('ROC plots t=%d' % t)

        fig1, ax1 = plt.subplots()
        roc_plot_setup(ax1, t, runid, 'train')

        fig2, ax2 = plt.subplots()
        roc_plot_setup(ax2, t, runid, 'test')

        for i in sm.folds.keys():
            roc_plot(ax1, sm, i, t, 'train')
            roc_plot(ax2, sm, i, t, 'test')

        roc_plot(ax1, sm, 'all', t, 'train')
        ax1.legend(loc=4, fontsize=10)
        plt.savefig('%s - ROC - Train - %d.png' % (runid, t), dpi=200)
        if False:
            plt.show()
        else:
            plt.close()

        roc_plot(ax2, sm, 'all', t, 'test')
        ax2.legend(loc=4, fontsize=10)
        plt.savefig(runid + '/' + runid + ' - ROC - Test - %d.png' % t, dpi=200)

        if False:
            plt.show()
        else:
            # wipe the plot cache to not use up extra memory
            plt.close()

    # metrics versus time plots
    ptf('Metrics vs time plots', logfile)
    alldf_test = sm.trigger_scores_test[sm.trigger_scores_test['fold'] == 'all']
    alldf_train = sm.trigger_scores[sm.trigger_scores['fold'] == 'all']
    metric_vs_time(alldf_train, runid, 'Train', debug=debug)
    metric_vs_time(alldf_test, runid, 'Test', debug=debug)

    # tmax vs t kde plots
    # tmax vs time in df_argmax
    # maxvalue of y', y'' in df_max
    df_max, df_argmax = make_df(sm, y, logfile)

    # add labels
    df_max = df_max.merge(right=y, how='outer', on='trial')
    df_argmax = df_argmax.merge(right=y, how='outer', on='trial')

    ch = make_ch(sm)
    for spot in ch:
        if spot[-1] == '1': # only look at first derivative
            # All together
            make_kde_plot(df_max, spot, runid,
                title='max yp', cmap='Greens', plotclass='All',
                logfile=logfile, debug=debug)

            make_kde_plot(df_argmax, spot, runid,
                title='tmax yp', cmap='Greens', plotclass='All',
                logfile=logfile, debug=debug)

            # detection specifying colors
            df = df_max[df_max['detection'] == 1]
            make_kde_plot(df, spot, runid,
                title='max yp', cmap='Reds', plotclass='D=1',
                logfile=logfile, debug=debug)

            df = df_argmax[df_argmax['detection'] == 1]
            make_kde_plot(df, spot, runid,
                title='tmax yp', cmap='Greens', plotclass='D=1',
                logfile=logfile, debug=debug)

            # detection controls
            df = df_max[df_max['detection'] == 0]
            make_kde_plot(df, spot, runid,
                title='max yp', cmap='Blues', plotclass='Control',
                logfile=logfile, debug=debug)

            df = df_argmax[df_argmax['detection'] == 0]
            make_kde_plot(df, spot, runid,
                title='tmax yp', cmap='Blues', plotclass='Control',
                logfile=logfile, debug=debug)

            # gram specifying colors
            df = df_argmax[df_argmax['gram'] == 'p']
            make_kde_plot(df, spot, runid,
                title='tmax yp', cmap='Purples', plotclass='P',
                logfile=logfile, debug=debug)

            df = df_argmax[df_argmax['gram'] == 'p']
            make_kde_plot(df, spot, runid,
                title='tmax yp', cmap='Oranges', plotclass='N',
                logfile=logfile, debug=debug)

            # df = df_argmax[df_argmax['gram'] == 'Controls']
            # make_kde_plot(df, spot, runid,
            #     title='tmax yp', cmap='Blues', plotclass='Control',
            #     logfile=logfile, debug=debug)

            # classification hoping for cyclical colors
            for species in sm.confusion_labels['classification']:
                if species == 'Control':
                    continue
                df = df_argmax[df_argmax['classification'] == species]

                # cycles thru colormap palettes
                cmap = sns.cubehelix_palette(light=1, as_cmap=True)

                make_kde_plot(df, spot, runid,
                    title='tmax yp', cmap=cmap, plotclass=species,
                    logfile=logfile, debug=debug)

        else:
            pass

        # plt.show()
    return df_max, df_argmax, ch

def make_ch(sm):
    # make channel headers
    ch = []
    ch = [[x+'1', x+'2'] for x in sm.trigger_spots]
    ch.append(['trial', 'timestep'])
    ch = np.array(ch).flatten()
    return ch

def index_to_trial_number(df, trial_hash):
    df['trial'].apply(lambda x: trial_hash[x])
    return df

def make_trial_hash(y):
    return {i:x for i,x in enumerate(y.index.values)}

def make_df(sm, y, runid, logfile=None):
    trial_hash = make_trial_hash(y)

    ch = make_ch(sm)
    # set-up data frame
    sm.logfile = logfile
    tf_collection = pd.DataFrame(columns=ch)
    tft_collection = pd.DataFrame(columns=ch)

    # append to dataframes
    index_list = []
    for t in sm.times:
        if t==12:
            continue
        tf = sm.load_time_step('trigger_features', t=t)
        tft = sm.load_time_step('trigger_feature_times', t=t)
        # check for nans
        # x = [i for i,k in enumerate(tf) if np.isnan(k)]
        if np.sum(np.isnan(tf)):
            ptf('NANS in timestep %d for features' % t, logfile)
            ptf(x, logfile)

        # x = [i for i,k in enumerate(tft) if np.isnan(k)]
        if np.sum(np.isnan(tft)):
            ptf('NANS in timestep %d for feature_timess' % t, logfile)
            ptf(x, logfile)



        # print t, tf.shape, tft.shape
        df = pd.DataFrame(tf, columns=ch[:-2])
        df['timestep'] = t
        df['trial'] = df.index
        if len(index_list):
            df['trial'] = index_list
        else:
            df['trial'] = df['trial'].apply(lambda x: trial_hash[x])
            index_list = df['trial'].values
        tf_collection = tf_collection.append(df)


        df = pd.DataFrame(tft, columns=ch[:-2])
        df['timestep'] = t
        df['trial'] = index_list
        tft_collection = tft_collection.append(df)


    # drop nans
    tf_collection = tf_collection.dropna()
    tft_collection = tft_collection.dropna()
    return tf_collection, tft_collection


def one_trial_vals(df1, trial, spot, type=1):
    df1 = df1[df1['trial']==trial]

    t = df1['timestep'].values
    # convert to hours from index
    t = t*20./60.

    y = df1[spot].values
    # convert to hours from minutes
    y = y/60.
    return t, y


def make_kde_plot(df, spot, runid, title=None, cmap='Greens', plotclass=None, logfile=None, debug=False):
    plt.figure()
    ptf('Plot KDE %s - %s' % (title, spot), logfile)
    x,y = stack_rows(df, spot)
    ptf('%s, %s' % (x.shape, y.shape), logfile)
    ptf('Check for nans', logfile)
    ptf('%s, %s' % (np.sum(np.isnan(x)), np.sum(np.isnan(y))), logfile)
    ptf('computing kde...', logfile)
    sns.kdeplot(x,y, shade=True, cmap=cmap)

    plottitle = runid + '-' + spot + ' - KDE trigger vs t'
    if title:
        plottitle += ' - ' + title
    if plotclass:
        plottitle += ' - ' + plotclass

    plt.title(plottitle)
    plt.xlabel('t (hrs)')
    if title:
        plt.ylabel(title)
    else:
        plt.ylabel('trigger metric')
    filename = runid + '/' + runid + '-' + spot + ' - KDE trigger vs t'
    if title:
        filename += ' - ' + title
    if plotclass:
        filename += ' - ' + plotclass

    ptf('Saving plot %s' % filename, logfile)
    plt.savefig(filename, dpi=200)
    if debug:
        plt.show()
    else:
        plt.close()


def myhist_2d(df, spot, runid, debug=False, title=None):
    # ts, tmaxs = stack_rows(df, spot)

    # time in hours
    # t_edges = np.arange(13/3., 60/3.,1/3.) # use for both x and y
    #
    # h, x, y, p  = plt.hist2d(ts, tmaxs, bins=t_edges)
    # plt.imshow(h, origin = "lower", interpolation = "gaussian")
    # plt.title(spot)
    plt.figure()
    print('Plot KDE %s - %s' % (title, spot))
    x,y = stack_rows(df, spot)
    print x.shape, y.shape
    print 'Check for nans'
    print np.sum(np.isnan(x)), np.sum(np.isnan(y))
    print ('computing kde...')
    sns.kdeplot(x,y, shade=True, cmap='Purples')
    plt.title(spot + ' - ' + title)
    plt.xlabel('t (hrs)')
    if title:
        plt.ylabel(title)
    else:
        plt.ylabel('trigger metric')
    plt.savefig(runid + '/' + runid + '-' + spot + ' - KDE trigger vs t - ' + title, dpi=200)
    if debug:
        plt.show()

    # now plot by class
    plt.figure()
    fig, ax = plt.subplots(figsize=(10,10))
    dfp = df[df['gram'] == 'p']
    print('Plot KDE P %s - %s' % (title, spot))
    x,y = stack_rows(dfp, spot)
    print x.shape, y.shape
    print 'Check for nans'
    print np.sum(np.isnan(x)), np.sum(np.isnan(y))
    print ('computing kde...')
    sns.kdeplot(x,y, shade=True, cmap='Reds', label='p')
    plt.title(spot + ' - p - ' + title)
    plt.savefig(runid + '/' + runid + '-' + spot + ' - KDE trigger vs t - p - ' + title, dpi=200)
    if debug:
        plt.show()

    dfn = df[df['gram'] == 'n']
    print('Plot KDE N %s - %s' % (title, spot))
    x,y = stack_rows(dfn, spot)
    print x.shape, y.shape
    print 'Check for nans'
    print np.sum(np.isnan(x)), np.sum(np.isnan(y))
    print ('computing kde...')
    sns.kdeplot(x,y, shade=True, cmap='Greens', label='n')
    plt.title(spot + ' - n - ' + title)
    plt.savefig(runid + '/' + runid + '-' + spot + ' - KDE trigger vs t - n - ' + title, dpi=200)
    if debug:
        plt.show()


    dfc = df[df['gram'] == 'Control']
    print('Plot KDE C %s - %s' % (title, spot))
    x,y = stack_rows(dfc, spot)
    print x.shape, y.shape
    print 'Check for nans'
    print np.sum(np.isnan(x)), np.sum(np.isnan(y))
    print ('computing kde...')
    sns.kdeplot(x,y, shade=True, cmap='Blues', label='c')

    plt.title(spot + ' - by class - ' + title)
    plt.xlabel('t (hrs)')
    if title:
        plt.ylabel(title)
    else:
        plt.ylabel('trigger metric')
    plt.legend(loc=2)
    plt.savefig(runid + '/' + runid + '-' + spot + ' - KDE trigger vs t by class - ' + title, dpi=200)
    plt.show()


def plot_kde(df, spot, cmap='Greens', title=None):
    print('Plot KDE %s - %s' % (title, spot))
    x,y = stack_rows(df, spot)
    print x.shape, y.shape
    print 'Check for nans'
    print np.sum(np.isnan(x)), np.sum(np.isnan(y))
    # data = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    # print data.shape
    print ('computing kde...')

    # plt.sca(ax)
    sns.kdeplot(x,y, shade=True, cmap=cmap)
    return x, y


def stack_rows(df, spot):
    ts = np.array([])
    tmaxs = np.array([])
    nrows = len(df)
    for row in range(0, len(df)):
        if row%1000 == 0:
            print 'Stacking row %d of %d' % (row, nrows)
        t, tmax = one_trial_vals(df, row, spot)
        ts = np.hstack((ts, t))
        tmaxs = np.hstack((tmaxs, tmax))
    return ts, tmaxs

def plot_feature(df1, df2, spot, trial):
    df1 = df1[df1['trial']==trial]
    df2 = df2[df2['trial']==trial]

    t = df1['timestep'].values*20.0/60.0

    y1 = df1[spot]
    y2 = df2[spot]
    fig, ax1 = plt.subplots(figsize=(12,6))

    ax1.plot(t,y1, 'b', linewidth=3, alpha=0.5, label='max')
#     ax1.legend(loc=1)
    ax1.set_xlabel('t hrs')

    ax1.set_ylabel('max', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()

    ax2.plot(t,y2, 'r', linewidth=3, alpha=0.5, label='argmax')
#     ax2.legend(loc=2)
    ax2.set_ylabel('argmax', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.title('S:%s, T:%d' % (spot, trial))
    plt.show()
