import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np

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
        text = 'F1:\t\t%0.3f\nACC:\t%0.3f\nPREC:\t%0.3f\nREC:\t%0.3f\nAUC:\t%0.3f' % (df['f1'],
            df['overall_accuracy'],df['precision'], df['recall'], df['auc'])
        ax.text(0.20, 0.02, text, verticalalignment='bottom', horizontalalignment='left',
        fontsize=15)
    else:
        ax.plot(x,y, label=str(fold))

def roc_plot_setup(ax, t, runid, testtrain='test'):
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('%s - ROC - %s - %d' % (runid, testtrain, t))

def make_trigger_plots(sm, runid, debug=False):
    # roc plots
    for t in sm.times:
        fig1, ax1 = plt.subplots(figsize=(12,12))
        roc_plot_setup(ax1, t, runid, 'train')

        fig2, ax2 = plt.subplots(figsize=(12,12))
        roc_plot_setup(ax2, t, runid, 'test')

        for i in sm.folds.keys():
            roc_plot(ax1, sm, i, t, 'train')
            roc_plot(ax2, sm, i, t, 'test')

        roc_plot(ax1, sm, 'all', t, 'train')
        ax1.legend(loc=4, fontsize=10)
        plt.savefig('%s - ROC - Train - %d.png' % (runid, t))

        roc_plot(ax2, sm, 'all', t, 'test')
        ax2.legend(loc=4, fontsize=10)
        plt.savefig('%s - ROC - Test - %d.png' % (runid, t))

        if debug:
            plt.show()
