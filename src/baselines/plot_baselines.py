import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_baselines(labels, my_predictor, isolation_forest, oc_svm, autoencoder):
    '''
    Plot predictor's metrics in a bar chart

    Parameters:
        labels (list): name of metrics
        my_predictor (list): scores of novel predictor
        isolation_forest (list): scores of isolation forest
        oc_svm (list): scores of one-class support vector machine
        autoencoder (list): scores of autoencoder
    '''
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width/2, my_predictor, width, label='Novel Predictor')
    rects2 = ax.bar(x - width/2, isolation_forest, width, label='Isolation Forest')
    rects3 = ax.bar(x + width/2, oc_svm, width, label='OC SVM')
    rects4 = ax.bar(x + 3*width/2, autoencoder, width, label='Autoencoder')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), borderaxespad=0., ncol=4)

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
    autolabel(rects4)

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    # take values from log files
    # [my predictor, isolation forest, oc svm, autoencoder]
    labels = ['TPR', 'TNR']
    my_predictor = [0.89865, 0.95101]
    isolation_forest = [0.86486,  0.93792]
    oc_svm = [0.89189, 0.89997]
    autoencoder = [0.91892, 0.92000]

    plot_baselines(labels, my_predictor, isolation_forest, oc_svm, autoencoder)
