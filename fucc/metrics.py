from inspect import indentsize
import os
from scikitplot.helpers import cumulative_gain_curve
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay, roc_curve
import numpy as np
#import mlflow

#def get_mlflow():
#    return mlflow.get_tracking_uri()

def plot_cumulative_gain(y_true, y_proba, title_fontsize=15, text_fontsize=10):
    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, y_proba)
    
    # Best classifier
    #percentages, gains2 = cumulative_gain_curve(y_true, y_true, True)

    fig, ax = plt.subplots(1, 1)

    ax.set_title('Cumulative gains chart', fontsize=title_fontsize)

    ax.plot(percentages, gains1, lw=3, label='Class {}'.format(True))
    
    # Best classifier
    #ax.plot(percentages, gains2, lw=3, label='Class {}'.format('best'))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)
    
    return ax


def get_confusion_matrix(y_true, y_pred_proba, threshold):
    cm = confusion_matrix(y_true, y_pred_proba>threshold)
    return cm

def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def get_optimal_f1_cutoff(y_true, y_pred_proba): #, thresholds = np.arange(0,1, 0.001)):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = (2 * (precisions * recalls) / (precisions + recalls))
    ix = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[ix] 
    optimal_f1_score = f1_scores[ix] 

    # too slow code below
    #scores = [f1_score(y_true, to_labels(y_pred_proba, t)) for t in thresholds]
    #ix = np.argmax(scores)
    #optimal_threshold = thresholds[ix] 
    #optimal_f1_score = scores[ix] 
    
    return optimal_threshold, optimal_f1_score

def get_lift_score(y_true, y_pred_proba, percentile, threshold):
    
    cutoff = round((len(y_pred_proba)*percentile))
    ind = np.argsort(y_pred_proba)[::-1][:cutoff]
    y_pred_sorted = y_pred_proba[ind] > threshold
    y_true_sorted = y_true.values[ind]

    with_model = np.sum(y_true_sorted) / len(y_true_sorted)
    without_model = np.sum(y_true) / len(y_true)
    lift_score = with_model/without_model
    
    return lift_score

def get_threshold_and_cutoff_for_positives(y_true, y_pred_proba, number_of_positives = 1000):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # SKlearn documentation states that n_thresholds = len(np.unique(probas_pred)). 
    # Hence, each threshold will result in one extra observation being classified as positive.
    # The thresholds are ordered from low to high
    ind = np.min([len(thresholds), number_of_positives])
    threshold = thresholds[-ind]
    cutoff = number_of_positives

    
    #f = 1
    # find threshold which coincides with the prefered alarm rate
    #for threshold in thresholds:
     #   if (np.sum(y_pred_proba >= threshold) <= number_of_positives):
     #       #print(threshold)
    #        f = threshold
     #       break
    #
    #if f < 1:
    #    cutoff = np.where(thresholds == f)[0][0]
    #else:
    #    cutoff = None
    return threshold, cutoff

def get_partial_ap(y_true, y_pred_proba, number_of_positives=1000):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    _, cutoff = get_threshold_and_cutoff_for_positives(y_true, y_pred_proba, number_of_positives=number_of_positives)
    partial_ap = -np.sum(np.diff(recalls[cutoff:]) * np.array(precisions)[cutoff:-1])
    return partial_ap
    
def plot_partial_ap(y_true, y_pred_proba, number_of_positives=1000):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    _, cutoff = get_threshold_and_cutoff_for_positives(y_true, y_pred_proba, number_of_positives=number_of_positives)

    fig = plt.figure()
    plt.step(recalls[cutoff:], precisions[cutoff:], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score: AP={0:0.2f}'
        .format(-np.sum(np.diff(recalls[cutoff:]) * np.array(precisions)[cutoff:-1])))

    return fig

def plot_ap(y_true, y_pred_proba):
    # AP curve
    aps = average_precision_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision = aps, estimator_name = None)
    disp.plot()
    return disp.figure_
    
def plot_roc(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    disp = RocCurveDisplay(fpr = fpr, tpr = tpr)
    disp.plot()
    return disp.figure_


def calculate_revenues(y_true, y_pred_proba, tx_amounts, number_of_positives = 1000):
    threshold, _ = get_threshold_and_cutoff_for_positives(y_true, y_pred_proba, number_of_positives)
    y_pred = (y_pred_proba >= threshold)
    # turn y_true into boolean array
    y_true = (y_true == True)
    
    tp = (y_pred[y_true == True] == True).flatten()
    fn = (y_pred[y_true == True] == False).flatten()
    
    # revenue of true positives
    revenue_tp = (tp * tx_amounts[y_true]).sum()
    
    # revenue of false negatives
    revenue_fn = (fn * tx_amounts[y_true]).sum()
    
    # total value of all fraud cases
    revenue_p = tx_amounts[y_true].sum()
    
    return revenue_tp, revenue_fn, revenue_p


def get_true_positives_at(y_true, y_pred_proba, number_of_positives = 1000):
    threshold, _ = get_threshold_and_cutoff_for_positives(y_true, y_pred_proba, number_of_positives)
    cm = confusion_matrix(y_true, y_pred_proba>=threshold)
    return cm[1,1]
    
def log_performance(y_true, y_pred_proba, images_path, name,  log_with_mlflow=False, number_of_positives=300):

    performance_dict = {}

    # plot ap curve
    print("plot ap curve")
    fig = plot_ap(y_true, y_pred_proba)
    plt.savefig(os.path.join(images_path, '_'.join([str(name), 'PRAUCcurve.pdf'])))
    plt.savefig(os.path.join(images_path, '_'.join([str(name), 'PRAUCcurve.png'])))
    #if log_with_mlflow:
        #mlflow.log_artifact(os.path.join(images_path, '_'.join([str(name), 'PRAUCcurve.pdf'])))

    # plot roc curve
    print("plot ROC curve")
    fig = plot_roc(y_true, y_pred_proba)
    plt.savefig(os.path.join(images_path, '_'.join([str(name), 'ROCcurve.pdf'])))
    plt.savefig(os.path.join(images_path, '_'.join([str(name), 'ROCcurve.png'])))
    
    # Cumulative gains cart
    print("plot cgc curve")
    ax = plot_cumulative_gain(y_true, y_pred_proba)
    fig = plt.gcf()
    plt.savefig(os.path.join(images_path, '_'.join([str(name), 'CumulativeGainsChart.pdf'])))
    plt.savefig(os.path.join(images_path, '_'.join([str(name), 'CumulativeGainsChart.png'])))

    #if log_with_mlflow:
        #mlflow.log_artifact(os.path.join(images_path, '_'.join([str(name), 'CumulativeGainsChart.pdf'])))

    print("get_optimal_f1")
    optimal_threshold, optimal_f1_score = get_optimal_f1_cutoff(y_true, y_pred_proba)
    #if log_with_mlflow:
        #mlflow.log_metric('optimal_threshold', optimal_threshold)
        #mlflow.log_metric('f1_score', optimal_f1_score)
    
    performance_dict['optimal_threshold'] = optimal_threshold
    performance_dict['f1_score'] = optimal_f1_score

    # Lift scores
    #percentiles = [0.05, 0.01, 0.001]
    #for percentile in percentiles:
    #    lift_score = get_lift_score(y_true, y_pred_proba, percentile, optimal_threshold)
    #    if log_with_mlflow:
    #        mlflow.log_metric('lift_score_' + str(percentile), lift_score)
    #    performance_dict['lift_score_'+ str(percentile)] = lift_score
    
    # confusion matrix
    cm = get_confusion_matrix(y_true, y_pred_proba, optimal_threshold)
    #if log_with_mlflow:
        #mlflow.log_metric('TN', cm[0,0])
        #mlflow.log_metric('FP', cm[0,1])
        #mlflow.log_metric('FN', cm[1,0])
        #mlflow.log_metric('TP', cm[1,1])    
    
    performance_dict['confusion_matrix/TN'] = cm[0,0]
    performance_dict['confusion_matrix/FP'] = cm[0,1]
    performance_dict['confusion_matrix/FN'] = cm[1,0]
    performance_dict['confusion_matrix/TP'] = cm[1,1]
    

    # partial ap
    #partial_ap = get_partial_ap(y_true, y_pred_proba, number_of_positives)
    #if log_with_mlflow:
    #    mlflow.log_metric('partial_ap', partial_ap)
    
    #performance_dict['partial_ap'] = partial_ap

    #fig = plot_partial_ap(y_true, y_pred_proba, number_of_positives)
    #plt.savefig(os.path.join(images_path, '_'.join([str(name), 'PPRAUCcurve.pdf'])))
    #if log_with_mlflow:
    #    mlflow.log_artifact(os.path.join(images_path, '_'.join([str(name), 'PPRAUCcurve.pdf'])))



    # catched@1000
    positives_at = get_true_positives_at(y_true, y_pred_proba, number_of_positives)
    #if log_with_mlflow:
        #mlflow.log_metric('catched_' + str(number_of_positives), positives_at)

    performance_dict['catched_' + str(number_of_positives)] = positives_at

    return performance_dict
