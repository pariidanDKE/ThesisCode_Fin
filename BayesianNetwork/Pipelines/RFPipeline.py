import pandas as pd
from BayesianNetwork.Pipelines.BayesPipeline import setup_data, determine_selection_method
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate


def cross_validation(model, _X, _y, _cv=5):
    '''Function to perform 5 Folds Cross-Validation
     Parameters
     ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    _X: array
         This is the matrix of features.
    _y: array
         This is the target variable.
    _cv: int, default=5
        Determines the number of folds for cross-validation.
     Returns
     -------
     The function returns a dictionary containing the metrics 'accuracy', 'precision',
     'recall', 'f1' for both training set and validation set.


     This function was taken from the education article : ##https://www.section.io/engineering-education/how-to-implement-k-fold-cross-validation/


    '''
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=_X,
                             y=_y,
                             cv=_cv,
                             scoring=_scoring,
                             return_train_score=True)

    return {"Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean() * 100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean() * 100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }


def predict(data):
    """
    This method builds the RF and determines,returns the standard classification metrics.

    :param data: The data that was selected using the selction method
    :return: Standard classification metrics.
    """
    data1 = data.copy()
    label = data1['Label']
    data1.drop(['Label'], axis=1, inplace=True)

    rf = RandomForestClassifier()
    metrics = cross_validation(rf, data1.values, label.values, _cv=10)
    print(metrics)
    return metrics


def pipeline(nodes_per_block, use_corr, use_rand, log_accuracy=False):
    """
    This method represents the pipline used to build a RF, using a specified configuration
    :param nodes_per_block: Number of nodes from each block
    :param use_corr: Use correlation as a selection method
    :param use_rand: Use baseline/random as a selection method
    """
    data, _ = setup_data(nodes_per_block, use_corr, use_rand, True)
    metrics = predict(data)

    accuracy = metrics['Mean Validation Accuracy'] / 100
    precision = metrics['Mean Validation Precision']
    recall = metrics['Mean Validation Recall']
    f1score = metrics['Mean Validation F1 Score']

    if log_accuracy:
        selection_method = determine_selection_method(use_corr, use_rand)
        accuracy_df = pd.read_csv('../Logs/inference_metrics.csv', index_col=0)
        accuracy_row = [nodes_per_block, 'N/A', 'RF(corr)', 'N/A', 'N/A', accuracy,
                        precision, recall, f1score]
        accuracy_df = accuracy_df.append(pd.Series(accuracy_row, index=accuracy_df.columns), ignore_index=True)
        accuracy_df.to_csv('..\\Logs\\inference_metrics_fin.csv')


if __name__ == '__main__':
    pipeline(nodes_per_block=25, use_corr=True, use_rand=False)
