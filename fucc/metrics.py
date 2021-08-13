from scikitplot.helpers import cumulative_gain_curve
import matplotlib.pyplot as plt

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