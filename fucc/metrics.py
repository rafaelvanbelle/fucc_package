import os
from scikitplot.helpers import cumulative_gain_curve
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, average_precision_score, PrecisionRecallDisplay
import numpy as np
import mlflow

def get_mlflow():
    return mlflow.get_tracking_uri()

def plot_cumulative_gain(y_true, y_proba, title_fontsize=15, text_fontsize=10):
    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, y_proba, True)
    
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

def get_optimal_f1_cutoff(y_true, y_pred_proba, thresholds = np.arange(0,1, 0.001)):
    scores = [f1_score(y_true, to_labels(y_pred_proba, t)) for t in thresholds]
    ix = np.argmax(scores)
    optimal_threshold = thresholds[ix] 
    optimal_f1_score = scores[ix] 
    
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

    # find threshold which coincides with the prefered alarm rate
    for threshold in np.flipud(thresholds):
        if (np.sum(y_pred_proba > threshold) >= number_of_positives):
            #print(threshold)
            f = threshold
            break

    cutoff = np.where(thresholds == f)[0][0]
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

def get_true_positives_at(y_true, y_pred_proba, number_of_positives = 1000):
    threshold, _ = get_threshold_and_cutoff_for_positives(y_true, y_pred_proba, number_of_positives)
    cm = confusion_matrix(y_true, y_pred_proba>=threshold)
    return cm[1,1]
    
def log_performance(y_true, y_pred_proba, images_path, name,  log_with_mlflow=False, number_of_positives=300):

    # plot ap curve
    fig = plot_ap(y_true, y_pred_proba)
    plt.savefig(os.path.join(images_path, '_'.join([str(name), 'PRAUCcurve.pdf'])))
    if log_with_mlflow:
        mlflow.log_artifact(os.path.join(images_path, '_'.join([str(name), 'PRAUCcurve.pdf'])))
    
    # Cumulative gains cart
    ax = plot_cumulative_gain(y_true, y_pred_proba)
    fig = plt.gcf()
    plt.savefig(os.path.join(images_path, '_'.join([str(name), 'CumulativeGainsChart.pdf'])))
    if mlflow:
        mlflow.log_artifact(os.path.join(images_path, '_'.join([str(name), 'CumulativeGainsChart.pdf'])))

    optimal_threshold, optimal_f1_score = get_optimal_f1_cutoff(y_true, y_pred_proba)
    if log_with_mlflow:
        mlflow.log_metric('optimal_threshold', optimal_threshold)
        mlflow.log_metric('f1_score', optimal_f1_score)

    # Lift scores
    percentiles = [0.05, 0.01, 0.001]
    for percentile in percentiles:
        lift_score = get_lift_score(y_true, y_pred_proba, percentile, optimal_threshold)
        if log_with_mlflow:
            mlflow.log_metric('lift_score_' + str(percentile), lift_score)
    
    # confusion matrix
    cm = get_confusion_matrix(y_true, y_pred_proba, optimal_threshold)
    if log_with_mlflow:
        mlflow.log_metric('TN', cm[0,0])
        mlflow.log_metric('FP', cm[0,1])
        mlflow.log_metric('FN', cm[1,0])
        mlflow.log_metric('TP', cm[1,1])    
    

    # partial ap
    partial_ap = get_partial_ap(y_true, y_pred_proba, number_of_positives)
    if log_with_mlflow:
        mlflow.log_metric('partial_ap', partial_ap)

    fig = plot_partial_ap(y_true, y_pred_proba, number_of_positives)
    plt.savefig(os.path.join(images_path, '_'.join([str(name), 'PPRAUCcurve.pdf'])))
    if log_with_mlflow:
        mlflow.log_artifact(os.path.join(images_path, '_'.join([str(name), 'PPRAUCcurve.pdf'])))


    # catched@1000
    positives_at = get_true_positives_at(y_true, y_pred_proba, number_of_positives)
    if log_with_mlflow:
        mlflow.log_metric('catched_' + str(number_of_positives), positives_at)
