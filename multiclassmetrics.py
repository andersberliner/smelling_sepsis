# multiclassmetrics.py
# Anders Berliner
# 20160108

from sklearn.metrics import confusion_matrix, classification_report
from itertools import izip
import numpy as np

def micro_average(func, results):
    total = np.zeros(4)
    # row of results is (TP,FP,FN,TN)
    for row in results:
        total += row
    # return func(total[0], total[1], total[2], total[3])
    return func(*total)


def macro_average(func, results):
    total = 0
    for row in results:
        total += func(*row)
    return total/float(len(results))


def accuracy_ovr(TP,FP,FN,TN):
    AC = 1.*(TP+TN)/np.sum([TP,FP,FN,TN])
    return AC

def precision_ovr(TP,FP,FN,TN):
    PR = 1.*TP/(TP+FP)
    return PR

def recall_ovr(TP,FP,FN,TN):
    RE = 1.*TP/(TP+FN)
    return RE

def f1_ovr(TP,FP,FN,TN):
    F1 = 2.*TP/(2*TP + FP + FN)
    return F1

def confusion_matrix_ovr(yt,yp,labels):
    cm = confusion_matrix(yt,yp,labels=labels)
    return cm.T

def results_ovr(yt,tp,labels):
    cm = confusion_matrix_ovr(yt,yp,labels)

    # Find TP, FP, FN, TN for each label
    results = np.zeros((len(labels), 4))
    total = len(yt)
    if len(yt) != np.sum(cm):
        print 'Error in finding confusion matrix'
    for i, label in enumerate(labels):
        TP = cm[i,i]
        FP = np.sum(cm[i,:]) - TP
        FN = np.sum(cm[:,i]) - TP
        TN = total - TP - FP - FN

        results[i,:] = np.array([TP,FP,FN,TN])
    return results

def scores_ovr(yt,yp,labels):
    # Calculate individual metrics - Accuracy, Precision, Recall, F1
    results = results_ovr(yt,yp,labels)
    scores = np.zeros((len(labels),4))
    total = len(yt)
    for i, (label, (TP,FP,FN,TN)) in enumerate(izip(labels, results)):
        AC = accuracy_ovr(TP,FP,FN,TN)
        PR = precision_ovr(TP,FP,FN,TN)
        RE = recall_ovr(TP,FP,FN,TN)
        F1 = f1_ovr(TP,FP,FN,TN)
        # AC = 1.*(TP+TN)/total
        # PR = 1.*TP/(TP+FP)
        # RE = 1.*TP/(TP+FN)
        # F1 = 2.*TP/(2*TP + FP + FN)

        scores[i,:] = np.array([AC,PR,RE,F1])

    return scores

def classification_report_ovr(yt,yp,labels, s1=11, s2=10):
    results = results_ovr(yt,yp,labels)
    scores = scores_ovr(yt,yp,labels)
    macros = [macro_average(accuracy_ovr, results),
              macro_average(precision_ovr, results),
              macro_average(recall_ovr, results),
              macro_average(f1_ovr, results)]
    micros = [micro_average(accuracy_ovr, results),
              micro_average(precision_ovr, results),
              micro_average(recall_ovr, results),
              micro_average(f1_ovr, results)]
    report = '%s %s %s %s %s %s' % (' '.rjust(s1),
                                    'precision'.rjust(s2),
                                    'recall'.rjust(s2),
                                    'f1-score'.rjust(s2),
                                    'support'.rjust(s2),
                                    'Accuracy'.rjust(s2))
    for i, label in enumerate(labels):
        AC = '%.3f' % scores[i,0]
        PR = '%.3f' % scores[i,1]
        RE = '%.3f' % scores[i,2]
        F1 = '%.3f' % scores[i,3]
        SU = '%d' % np.sum(results[0,i] + results[2,i])
        report += '\n%s %s %s %s %s %s' % (str(label).rjust(s1),
                                           PR.rjust(s2),
                                           RE.rjust(s2),
                                           F1.rjust(s2),
                                           SU.rjust(s2),
                                           AC.rjust(s2))

    AC = '%.3f' % micros[0]
    PR = '%.3f' % micros[1]
    RE = '%.3f' % micros[2]
    F1 = '%.3f' % micros[3]
    TO = str(len(yt))
    report += '\n'
    report += '\n%s %s %s %s %s %s' % ('Micro avg'.rjust(s1),
                                             PR.rjust(s2),
                                             RE.rjust(s2),
                                             F1.rjust(s2),
                                             TO.rjust(s2),
                                             AC.rjust(s2))

    AC = '%.3f' % macros[0]
    PR = '%.3f' % macros[1]
    RE = '%.3f' % macros[2]
    F1 = '%.3f' % macros[3]
    report += '\n%s %s %s %s %s %s' % ('Macro avg'.rjust(s1),
                                                PR.rjust(10),
                                                RE.rjust(10),
                                                F1.rjust(10),
                                                TO.rjust(10),
                                                AC.rjust(10))

    return report

#
# beer_recs = beers[beers['ID']
#                .isin(pred)] \
#                .to_html(columns=['Name', 'Style', 'Brewery Name'],index=False) \
#                .replace('border="1" class="dataframe"','class=table table-hover')


if __name__ == '__main__':
# play around with these metrics - 10 each of 1, 2 and 3
    yt = [1,1,1,1,1,1,1,1,1,1,
          2,2,2,2,2,2,2,2,2,2,
          3,3,3,3,3,3,3,3,3,3]
    yp = [1,1,1,1,1,1,2,2,3,3,
          2,2,2,2,2,1,1,3,3,3,
          3,3,3,3,3,3,3,3,1,2]

    cm = confusion_matrix_ovr(yt,yp,[1,2,3])
    results = results_ovr(yt,yp,[1,2,3])
    scores = scores_ovr(yt,yp,[1,2,3])

    print 'See results for toy dataset since sklearn is unreliable'
    print 'SKLEARN confusion matrix, transposed'
    print np.transpose(confusion_matrix(yt,yp, [1,2,3]))
    print 'My confusion matrix'
    print cm
    print 'Accuracy, Precision, Recall, F1-score'
    print scores
    print 'SKLEARN classification report'
    print classification_report(yt,yp,[1,2,3])
    print 'My classification report'
    print classification_report_ovr(yt,yp,[1,2,3])
