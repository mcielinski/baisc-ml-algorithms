import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Describe data
def describe(data):
    transpose = data.transpose()
    for column in transpose:
        print(stats.describe(column))


# Plot data
def plot_dataset(dataset, attr='Class'):
    columns = dataset.drop(columns=[attr]).columns
    fig = sns.FacetGrid(dataset, hue=attr,  height=5, palette="bright") \
        .map(plt.scatter, columns[0], columns[1]).add_legend()


# Data Pre-processing
def normalize(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)

def standardize(data):
    return preprocessing.scale(data)


# Metrics
round_precision = 3

def get_accuracy(y_real, y_pred):
    return round(accuracy_score(y_real, y_pred), round_precision)

def get_precision(y_real, y_pred):
    return round(precision_score(y_real, y_pred, average='macro'), round_precision)

def get_recall(y_real, y_pred):
    return round(recall_score(y_real, y_pred, average='macro'), round_precision)

def get_FSC(y_real, y_pred):
    return round(f1_score(y_real, y_pred, average='macro'), round_precision)

def getAllMetrics(y_real, y_pred):
    print("Accuracy:", get_accuracy(y_real, y_pred))
    print("Precision:", get_precision(y_real, y_pred))
    print("Recall:", get_recall(y_real, y_pred))
    print("F1 score:", get_FSC(y_real, y_pred))


# Bar plot
def bar_plot(name, plot_data, out_path, to_file='', show=True, save=False):
    bars = plot_data.keys()
    height = plot_data.values()
    
    y_pos = np.arange(len(bars))
    plt.ylim(0,1)
    plt.title(name)
    plt.bar(y_pos, height, color=['red', 'green', 'blue']) #, alpha=0.2)
    plt.xticks(y_pos, bars)
    #bars.set_rasterized(True)
    
    if show:
        plt.show()
    elif save:
        plt.savefig('{}{}_{}.png'.format(out_path, name, to_file), dpi=120)