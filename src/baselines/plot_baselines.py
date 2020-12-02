import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_baselines(labels, predictor, isolation_forest, oc_svm):
    '''
    Plot predictor's metrics in a bar chart

    Parameters:
        labels (list): name of metrics
        predictor (list): scores of novel predictor
        isolation_forest (list): scores of isolation forest
        oc_svm (list): scores of one-class support vector machine
    '''
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, predictor, width, label='Predictor')
    rects2 = ax.bar(x, isolation_forest, width, label='Isolation Forest')
    rects3 = ax.bar(x + width, oc_svm, width, label='OC SVM')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), borderaxespad=0., ncol=3)

    def autolabel(rects):
    # Attach a text label above each bar in *rects*, displaying its height.
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.4f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    # take values from log files
    # [predictor, isolation forest, oc svm]
    labels = ['TPR', 'TNR', 'ROC AUC']
    predictor = [0.87602, 0.96544, 0.92073]
    isolation_forest = [ 0.83333,  0.96553, 0.89943]
    oc_svm = [0.88618, 0.89988, 0.89303]

    plot_baselines(labels, predictor, isolation_forest, oc_svm)
