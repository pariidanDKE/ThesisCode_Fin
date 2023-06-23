import pandas as pd
import numpy as np


def discretize_3tiers(data, learn_network, TIERS_NUM=3):
    """
    the code discretized data was taken from an example in the documentation of pgmpy
    https://pgmpy.org/detailed_notebooks/11.%20A%20Bayesian%20Network%20to%20model%20the%20influence%20of%20energy%20consumption%20on%20greenhouse%20gases%20in%20Italy.html

    """
    def boundary_str(start, end, tier):
        return f'{tier}: {start:+0,.2f} to {end:+0,.2f}'

    def relabel(v, boundaries):
        for index, (start, end) in enumerate(boundaries):
            if v >= start and v <= end:
                tier = chr(ord('A') + index)
                return boundary_str(start, end, tier=tier)
        return np.nan

    def get_boundaries(tiers):
        prev_tier = tiers[0]
        boundaries = [(prev_tier[0], prev_tier[prev_tier.shape[0] - 1])]
        for index, tier in enumerate(tiers):
            if index != 0:
                boundaries.append((prev_tier[prev_tier.shape[0] - 1], tier[tier.shape[0] - 1]))
                prev_tier = tier
        return boundaries

    new_columns = {}
    if learn_network:
        tmp = data.drop(['Label'], axis=1)
    else:
        tmp = data
    for i, content in enumerate(tmp.items()):
        (label, series) = content
        values = np.sort(np.array([x for x in series.tolist() if not np.isnan(x)], dtype=float))
        if values.shape[0] < TIERS_NUM:
            print(f'Error: there are not enough data for label {label}')
            break
        boundaries = get_boundaries(tiers=np.array_split(values, TIERS_NUM))
        new_columns[label] = [relabel(value, boundaries) for value in series.tolist()]

    df = pd.DataFrame(data=new_columns)
    df.columns = tmp.columns
    df.index = tmp.index

    if learn_network:
        df['Label'] = data['Label']

    return df


def discretize_entropy(data, learn_network):
    # from scipy.stats import entropy
    from collections import Counter
    import math
    def gini_impurity(y):
        """
        Given a Pandas Series, it calculates the Gini Impurity.
        y: variable with which calculate Gini Impurity.
        """
        if isinstance(y, pd.Series):
            p = y.value_counts() / y.shape[0]
            gini = 1 - np.sum(p ** 2)
            return gini

        else:
            raise 'Object must be a Pandas Series.'

    def entropy(y):
        '''
        Given a Pandas Series, it calculates the entropy.
        y: variable with which calculate entropy.
        '''
        if isinstance(y, pd.Series):
            a = y.value_counts() / y.shape[0]
            entropy = np.sum(-a * np.log2(a + 1e-9))
            return (entropy)

        else:
            raise 'Object must be a Pandas Series.'

    def variance(y):
        '''
        Function to help calculate the variance avoiding nan.
        y: variable to calculate variance to. It should be a Pandas Series.
        '''
        if len(y) == 1:
            return 0
        else:
            return y.var()

    def information_gain(y, mask, func=entropy):
        '''
        It returns the Information Gain of a variable given a loss function.
        y: target variable.
        mask: split choice.
        func: function to be used to calculate Information Gain in case os classification.
        '''

        a = sum(mask)
        b = mask.shape[0] - a

        if a == 0 or b == 0:
            ig = 0

        else:
            if y.dtypes != 'O':
                ig = variance(y) - (a / (a + b) * variance(y[mask])) - (b / (a + b) * variance(y[-mask]))
            else:
                ig = func(y) - a / (a + b) * func(y[mask]) - b / (a + b) * func(y[-mask])

        return ig

    import itertools

    def categorical_options(a):
        '''
        Creates all possible combinations from a Pandas Series.
        a: Pandas Series from where to get all possible combinations.
        '''
        a = a.unique()

        opciones = []
        for L in range(0, len(a) + 1):
            for subset in itertools.combinations(a, L):
                subset = list(subset)
                opciones.append(subset)

        return opciones[1:-1]

    def max_information_gain_split(x, y, func=entropy):
        '''
        Given a predictor & target variable, returns the best split, the error and the type of variable based on a selected cost function.
        x: predictor variable as Pandas Series.
        y: target variable as Pandas Series.
        func: function to be used to calculate the best split.
        '''

        split_value = []
        ig = []

        numeric_variable = True if x.dtypes != 'O' else False

        # Create options according to variable type
        if numeric_variable:
            options = x.sort_values().unique()[1:]
        else:
            options = categorical_options(x)

        # Calculate ig for all values
        for val in options:
            mask = x < val if numeric_variable else x.isin(val)
            val_ig = information_gain(y, mask, func)
            # Append results
            ig.append(val_ig)
            split_value.append(val)

        # Check if there are more than 1 results if not, return False
        if len(ig) == 0:
            return None, None, None, False

        else:
            # Get results with highest IG
            best_ig = max(ig)
            best_ig_index = ig.index(best_ig)
            best_split = split_value[best_ig_index]
            return best_ig, best_split, numeric_variable, True

    split_df = data.drop('Label', axis=1).apply(max_information_gain_split, y=data['Label'])
    data_cat = data.copy()
    for col in data.drop('Label', axis=1).columns:
        data_cat[col] = np.where(data[col] > split_df[col][1], 'A', 'B')
    return data_cat


def discretize_data(data, use_3tiers, learn_network):
    if use_3tiers:
        discrete_data = discretize_3tiers(data, learn_network)
    elif learn_network:
        discrete_data = discretize_entropy(data, learn_network)
        discrete_data['Label'] = data['Label']
    return discrete_data