import pandas as pd
import itertools
import numpy as np
import requests
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from pgmpy.estimators import HillClimbSearch, PC
from pgmpy.estimators import K2Score
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianNetwork
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from random import sample
import warnings

from DicretizeData import discretize_data
warnings.filterwarnings('ignore')


def setup_data(nodes_per_block, use_corr, use_rand, learn_network):
    """
    This method performs the initial setup for the entire pipeline.

    :param nodes_per_block: nodes chosen from each omic type
    :param use_corr:  use correlation selection method
    :param use_rand:  use baseline selection method
    :param learn_network: if learn_network, add the Label in the dataframes
    :return: the merged df, the variables chosen by the selection method
    """
    df_mir = pd.read_csv('../Data/mir_data_labelled.csv', index_col=0)
    df_pr = pd.read_csv('../Data/pr_data_labelled.csv', index_col=0)
    df_mrna = pd.read_csv('../Data/mrna_data_labelled.csv', index_col=0)

    if use_corr:
        top_loadings = pd.read_csv('../Data/BayesVariables_Allblocks_CorrelationOnly.csv', index_col=0)
    elif use_rand:
        top_loadings = random_sample(df_pr, df_mir, df_mrna, nodes_per_block)
    elif learn_network:
        top_loadings = pd.read_csv(
            '../Data/BayesVariables_Allblocks_mRNACorrelation.csv')
    else:
        top_loadings = pd.read_csv(
            '../Data/BayesVariables_Allblocks_OnlyPC1.csv')

    top_loadings['microRNA'] = top_loadings['microRNA'].apply(str.lower)
    df_mir.columns = [x.lower() for x in df_mir.columns]

    df_mir_top = df_mir[top_loadings['microRNA'].head(nodes_per_block)]
    df_pr_top = df_pr[top_loadings['Protein'].head(nodes_per_block)]
    df_mrna_top = df_mrna[top_loadings['mRNA'].head(nodes_per_block)]

    data = df_mir_top.join(df_mrna_top).join(df_pr_top)

    if learn_network:
        data['Label'] = df_mir['braak']
        data['Label'] = data['Label'].replace('Case', 1)
        data['Label'] = data['Label'].replace('Control', 0)

    return data, top_loadings


def random_sample(df_pr, df_mir, df_mrna, n=10):
    """
    This method performs the Random Selection, choosing n random variables from each block.

    :param df_pr: proten data
    :param df_mir:  microRNA data
    :param df_mrna: mrna data
    :param n:  number of variables picked, per block
    :return: the variables that were picked, n per block
    """
    top_loadings = pd.DataFrame(columns=['microRNA', 'Protein', 'mRNA'])

    top_loadings['Protein'] = sample(df_pr.columns.tolist(), n)
    top_loadings['microRNA'] = sample(df_mir.columns.tolist(), n)
    top_loadings['mRNA'] = sample(df_mrna.columns.tolist(), n)
    return top_loadings


def compute_blacklist(top_loadings, nodes_per_block):
    """
    This method is a helper method, it computes the blacklist for the HillClimb method, as in edges that are not allowed.

    :param top_loadings: selected variables from each block
    :param nodes_per_block: number of nodes per block
    :return: the computed black_list
    """
    black_list = list(itertools.permutations(top_loadings['microRNA'].head(nodes_per_block).values, 2))
    black_list.extend(list(itertools.permutations(top_loadings['Protein'].head(nodes_per_block).values, 2)))
    black_list.extend(list(itertools.permutations(top_loadings['mRNA'].head(nodes_per_block).values, 2)))
    black_list.extend(list(itertools.product(['Label'], top_loadings['microRNA'].head(nodes_per_block))))
    black_list.extend(list(itertools.product(['Label'], top_loadings['Protein'].head(nodes_per_block))))
    black_list.extend(list(itertools.product(['Label'], top_loadings['mRNA'].head(nodes_per_block))))
    return black_list


def compute_whitelist(top_loadings, nodes_per_block):
    """
    This method is a helper method, it computes the whitelist for the HillClimb method, as in the defined edge search space.

    :param top_loadings: selected variables from each block
    :param nodes_per_block: number of nodes per block
    :return:
    """
    white_list = list(
        itertools.product(top_loadings['microRNA'].head(nodes_per_block),
                          top_loadings['Protein'].head(nodes_per_block)))

    white_list.extend(
        list(itertools.product(top_loadings['microRNA'].head(nodes_per_block),
                               top_loadings['mRNA'].head(nodes_per_block))))

    white_list.extend(
        list(itertools.product(top_loadings['mRNA'].head(nodes_per_block),
                               top_loadings['Protein'].head(nodes_per_block))))
    white_list.extend(list(itertools.product(top_loadings['Protein'].head(nodes_per_block), ['Label'])))
    white_list.extend(list(itertools.product(top_loadings['mRNA'].head(nodes_per_block), ['Label'])))
    white_list.extend(list(itertools.product(top_loadings['microRNA'].head(nodes_per_block), ['Label'])))
    return white_list


def compute_fixed_edges(top_loadings, fixed_edges_count=2):
    """
    This method is a helper method, it computes the fixed_list, as in edges that need to be in the network. It picks top n protein, and forces interaction with label.
    This method is not used, since it requires too much computational power.

    :param top_loadings: selected variables from each block
    :param fixed_edges_count: number of fixed edges
    :return:
    """
    fixed_edges = list(itertools.product(top_loadings['Protein'].head(fixed_edges_count), ['Label']))
    return fixed_edges


def structure_learning(data, top_loadings, nodes_per_block, use_pc=False, use_black=True, use_white=True,
                       use_fixed_edges=False):
    """
    This method computed the model, based on the specified configuration.

    :param data: the merged dataframe
    :param top_loadings:  selected variables
    :param nodes_per_block: number of nodes per block
    :param use_pc: use the PC structure learning algorithm
    :param use_black: define a blacklist for HillClimb
    :param use_white: define a whitelist for HillClimb
    :param use_fixed_edges: define fixed edges for HillClimb (not in use)
    :return: The computed model
    """
    if use_pc:
        sl = PC(data)
    else:
        sl = HillClimbSearch(data)

    black_list = compute_blacklist(top_loadings, nodes_per_block)
    white_list = compute_whitelist(top_loadings, nodes_per_block)
    fixed_edges = compute_fixed_edges(top_loadings, nodes_per_block)

    data = data.sample(frac=1, axis=0)

    if use_black and use_white and use_fixed_edges:
        best_model = sl.estimate(scoring_method=K2Score(data), white_list=white_list, black_list=black_list,
                                 fixed_edges=fixed_edges)
    elif not use_fixed_edges:
        if use_black and use_white:
            best_model = sl.estimate(scoring_method=K2Score(data), white_list=white_list, black_list=black_list)
        elif use_black:
            best_model = sl.estimate(scoring_method=K2Score(data), black_list=black_list)
        elif use_white:
            best_model = sl.estimate(scoring_method=K2Score(data), white_list=white_list)
        else:
            best_model = sl.estimate(scoring_method=K2Score(data))
    else:
        best_model = sl.estimate(scoring_method=K2Score(data),
                                 fixed_edges=fixed_edges, significance_level=0.2, variant='orig')

    print(best_model)
    return best_model


def visualize_network(model, data):
    """
    This method visualized the network, using interactive physics.
    method taken from https://towardsdatascience.com/visualizing-protein-networks-in-python-58a9b51be9d5

    :param model: Computed model
    :param data: Merged data
    """
    net = Network(directed=True)
    net.add_nodes(data.columns.values)
    net.add_edges(model.edges)
    # net.nodes
    net.show(name='nx.html', notebook=False)


def fit_network(data, edges):
    """
    This method finds the CPDs of the nodes, and computes classification metrics for predicting Label.

    :param data: Merged data
    :param edges: Edges of network
    :return: Classification Metrics
    """
    nodes = list(set(itertools.chain.from_iterable(edges)))
    data_cat = data[nodes]
    kf = KFold(n_splits=5)
    kf.get_n_splits(data_cat)

    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
    for i, (train_index, test_index) in enumerate(kf.split(data_cat)):
        model = BayesianNetwork(edges)

        train_data = data_cat.loc[data_cat.index[train_index]]
        predict_data = data_cat.loc[data_cat.index[test_index]]
        y_test = predict_data['Label'].copy()

        model.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)
        model.get_cpds('Label')

        predict_data = predict_data.copy()

        predict_data.drop('Label', axis=1, inplace=True)
        y_pred = model.predict(predict_data)

        accuracies.append(accuracy_score(y_test.values, y_pred.values))
        precisions.append(precision_score(y_test.values, y_pred.values))
        recalls.append(recall_score(y_test.values, y_pred.values))
        f1scores.append(f1_score(y_test.values, y_pred.values))

    cv_accuracy = np.mean(accuracies)
    cv_precision = np.mean(precisions)
    cv_recall = np.mean(recalls)
    cv_f1score = np.mean(f1scores)

    print("CV Accuracy:", cv_accuracy, '; for model with ', len(nodes), ' nodes and ', len(edges), ' edges.')

    metrics = [cv_accuracy, cv_precision, cv_recall, cv_f1score]
    print(metrics)
    return metrics


def filter_mRNA(top_loadings, top_gene_count=5):
    """
    This method imports microRNA interactions, which are downloaded from miRTaRBase, and configures them in the proper edge format.

    :param top_loadings: selected variables
    :param top_gene_count: Number of targeted genes that are selected for each microRNA
    :return: Edges formed by microRNAs and their targeted edges
    """
    mir_interactions_df = pd.read_csv('../Data/mir_interactions_onlyhsa.csv')

    merged_df = top_loadings.merge(mir_interactions_df, right_on='miRNA', left_on='microRNA')
    merged_df['Experiments'] = merged_df['Experiments'].apply(str.lower)

    strong_evidence_df = merged_df[(merged_df['Experiments'].str.contains("reporter")) |
                                   (merged_df['Experiments'].str.contains("qrt")) |
                                   (merged_df['Experiments'].str.contains("western"))
                                   ]

    strong_evidence_df = strong_evidence_df[['microRNA', 'Target Gene']].drop_duplicates()

    microRNA_TopGenes = pd.DataFrame().reindex_like(strong_evidence_df).dropna()
    for miRNA in top_loadings['microRNA'].values:
        microRNA_TopGenes = pd.concat(
            [microRNA_TopGenes, strong_evidence_df.loc[strong_evidence_df['microRNA'] == miRNA].head(top_gene_count)])
    return microRNA_TopGenes[['microRNA', 'Target Gene']]


def get_string_identifiers(model, microRNA_TopGenes):
    """
    This method finds the standard STRING names for all of the variables,by using the STRING API. returns a dataframe containing pairings of the original names and the STRING names.

    method inspired by  https://towardsdatascience.com/visualizing-protein-networks-in-python-58a9b51be9d5
    https://string-db.org/cgi/help?sessionId=bDpJUWrfijDt - api documentation

    :param model: Computed Network
    :param microRNA_TopGenes: The microRNA,Gene edges found in filter_mRNA method.
    :return: Dataframe containing pairings of the original names and the STRING names.
    """
    nodes = list(set(itertools.chain.from_iterable(model.edges)))
    protein_list = nodes

    for i in range(0, len(protein_list)):

        if 'mir' not in protein_list[i]:
            protein_list[i] = protein_list[i].partition("-")[0]

    proteins = '%0d'.join(protein_list)
    url = 'https://string-db.org//api/tsv/get_string_ids?identifiers=' + proteins + '&species=9606'
    r = requests.get(url)

    lines = r.text.split('\n')  # pull the text from the response object and split based on new lines
    data = [l.split('\t') for l in lines]  # split each line into its components based on tabs
    # convert to dataframe using the first row

    df_id = pd.DataFrame(data[1:-1], columns=data[0])

    found_proteins = [protein_list[int(q_idx)] for q_idx in list(df_id.queryIndex.values)]
    df_id['MyDataName'] = found_proteins

    df_mir_id = microRNA_TopGenes.rename(columns={"microRNA": "MyDataName", "Target Gene": "preferredName"})
    df_mir_id = df_mir_id[df_mir_id['preferredName'].apply(lambda x: isinstance(x, str))]

    if df_mir_id.size > 0:
        intersect = [value for value in df_mir_id['MyDataName'].tolist() if value in nodes]

        if len(intersect) > 0:
            df_id = pd.concat([df_id, df_mir_id[intersect]], ignore_index=True)

    return df_id.drop_duplicates()


def request_interactions(df_id):
    """
    This method finds interactions of the BNs variables, by using the STRING API. Returns these found interactions

    method inspired by  https://towardsdatascience.com/visualizing-protein-networks-in-python-58a9b51be9d5
    https://string-db.org/cgi/help?sessionId=bDpJUWrfijDt - api documentation

    :param df_id: Dataframe containing pairings of the original variable names  and the STRING names.
    :return: The interactions found between the nodes of the network.
    """
    identifiers = '%0d'.join(list(df_id['preferredName'].unique()[:500]))
    url = 'https://string-db.org/api/tsv/network?identifiers=' + identifiers + '&species=9606'
    r = requests.get(url)

    lines = r.text.split('\n')  # pull the text from the response object and split based on new lines
    data = [l.split('\t') for l in lines]  # split each line into its components based on tabs
    # convert to dataframe using the first row as the column names; drop empty, final row
    df = pd.DataFrame(data[1:-1], columns=data[0])

    identifiers2 = '%0d'.join(list(df_id['preferredName'].unique()[500:]))
    url2 = 'https://string-db.org/api/tsv/network?identifiers=' + identifiers2 + '&species=9606'
    r2 = requests.get(url2)

    lines2 = r2.text.split('\n')  # pull the text from the response object and split based on new lines
    data2 = [l.split('\t') for l in lines2]  # split each line into its components based on tabs
    # convert to dataframe using the first row as the column names; drop empty, final row
    df2 = pd.DataFrame(data2[1:-1], columns=data2[0])
    df = pd.concat([df, df2])

    # dataframe with the preferred names of the two proteins and the score of the interaction
    interactions = df[['preferredName_A', 'preferredName_B', 'score']]
    interactions.drop_duplicates(inplace=True)
    interactions.reset_index(drop=True, inplace=True)

    return interactions


def map_interactions(interactions, df_id):
    """
    This method maps the interactions from STRING back to the original names.

    :param interactions: 'Real' edges found
    :param df_id: Dataframe containing pairings of the original variable names  and the STRING names.
    :return: Interactions mapped to their original names
    """
    interactions_mapped = pd.DataFrame(columns=['from', 'to', 'score'])

    for i in range(1, len(interactions)):

        mapped_A = df_id.loc[df_id['preferredName'] == interactions.iloc[i]['preferredName_A']]
        mapped_B = df_id.loc[df_id['preferredName'] == interactions.iloc[i]['preferredName_B']]

        pairwise_all = list(
            itertools.product((list(mapped_A['MyDataName'].values)), (list(mapped_B['MyDataName'].values))))

        for pair in pairwise_all:
            if pair[0] == pair[1]:
                pairwise_all.remove(pair)

        pairwise_df = pd.DataFrame(pairwise_all, columns=['from', 'to', ])
        pairwise_df['score'] = interactions.iloc[i]['score']

        if len(mapped_A) > 0:
            interactions_mapped = pd.concat([interactions_mapped, pairwise_df])

    return interactions_mapped.sort_values(by='score', ascending=False)


def visualize_string_network(interactions, title):
    """
    This method receives labelled interactions, and displays a plot of the network

    :param interactions: Edges of the network
    :param title: Title of the plot
    """
    G = nx.Graph(name=title)
    interactions = np.array(interactions)
    for i in range(len(interactions)):
        interaction = interactions[i]
        a = interaction[0]  # protein a node
        b = interaction[1]  # protein b node
        w = 1  # score as weighted edge where high scores = low weight
        G.add_weighted_edges_from([(a, b, w)])  # add weighted edge to graph

    pos = nx.spring_layout(G)  # position the nodes using the spring layout
    plt.figure(figsize=(15, 15))
    nx.draw_networkx(G, arrows=True, font_size=8, node_size=200, with_labels=True)
    plt.axis('off')
    plt.show()


def process_interactions(interactions):
    """
    This is a helper method, it removes duplicates and converts the interactions into a list of tuples
    :param interactions: Edges
    :return: Processed Edges
    """
    edges = [tuple([i[0], i[1]]) for i in np.array(interactions)]
    edges = remove_duplicates_from_list_of_tuples(edges)

    return edges


def determine_match_count(edges_space, model, log_matching, nodes_per_block, selection_method, config,
                          structure_method):
    """
    This method determines how many matches there are between the computed edges and 'real' edges.

    :param edges_space: 'Real' interactions found using STRING
    :param model: The computed model
    :param log_matching: Write the found matching to matching_results_fin
    :param nodes_per_block: Number of nodes for each block
    :param selection_method: The selection method used during Structure Learning
    :param config: Whether whitelist,blacklist was used during Structure Learning
    :param structure_method: The Structure Learning Algorithm used
    :return: List of the interactions that were found in both computed and 'real' network.
    """
    real_edges = edges_space
    computed_edges = list(model.edges)

    matching_lists = [sublist for sublist in real_edges if sublist in computed_edges]

    print('There are', len(computed_edges), 'edges in our computed network')
    print('From nodes of our network, STRING found', len(real_edges), 'edges connecting them.')
    print('Found ' + str(len(matching_lists)) + ' matches between the two edge lists!')
    print(matching_lists)
    if log_matching:
        matching_df = pd.read_csv('../Logs/matching_results_fin.csv', index_col=0)

        matching_row = [nodes_per_block, len(computed_edges), len(model.nodes), len(edges_space),
                        len(set([item for sublist in edges_space for item in sublist])), matching_lists,
                        len(matching_lists), selection_method, config, structure_method]

        print(matching_row)
        matching_row_df = pd.DataFrame([matching_row], columns=matching_df.columns)
        matching_df = pd.concat([matching_df, matching_row_df])

        matching_df.to_csv('..\\Logs\\matching_results_fin.csv')

    return matching_lists


def compute_filenames(n, use_white, use_black, use_fixed_edges, use_corr, use_rand):
    """
    This is a helper method, specifies the file name based on the configuration of the Structure Learning.
    """
    edge_filename = '..\\BayesianModels\\edges_head' + str(n) + '_'
    node_filename = '..\\BayesianModels\\nodes_head' + str(n) + '_'

    if use_corr:
        edge_filename += '_corr_'
        node_filename += '_corr_'

    if use_rand:
        edge_filename += '_rand_'
        node_filename += '_corr_'

    if use_white:
        edge_filename += 'w'
        node_filename += 'w'
    if use_black:
        edge_filename += 'b'
        node_filename += 'b'
    if use_fixed_edges:
        edge_filename += 'fe'
        node_filename += 'fe'

    edge_filename += '.csv'
    node_filename += '.csv'

    return edge_filename, node_filename


def keep_needed_edges(data, edges):
    """
    This method is used as a part of the inference pipeline. It keeps only the connected structure that the node Label is part of.

    :param data: Merged data
    :param edges: Edges of the network.
    :return: The connected structure of which 'Label' is a part of.
    """
    edge_df = pd.DataFrame(edges, columns=['from', 'to'])

    needed_nodes = ['Label']
    edges = list(edges)

    i = 0
    while True:
        if i == len(edges):
            break
        if edges[i][0] in needed_nodes and edges[i][1] not in needed_nodes:
            # print(edges[i])
            needed_nodes.append(edges[i][1])
            i = 0
        elif edges[i][1] in needed_nodes and edges[i][0] not in needed_nodes:
            # print(edges[i])
            needed_nodes.append(edges[i][0])
            i = 0
        else:
            i += 1

    needed_edges = edge_df.loc[
        (edge_df['from'].isin(needed_nodes)) | (edge_df['to'].isin(needed_nodes))].values.tolist()
    needed_data = data[needed_nodes]

    return needed_data, needed_edges


def log_network_accuracy(metrics, model, nodes_per_block, selection_method, config):
    """
    This method writes the accuracy, as well as network properties to inference_metrics_fin.
    """
    accuracy_df = pd.read_csv('../Logs/inference_metrics_fin.csv', index_col=0)

    accuracy_row = [nodes_per_block, config, selection_method, len(model.nodes), len(model.edges), metrics[0],
                    metrics[1], metrics[2], metrics[3]]

    accuracy_row_df = pd.DataFrame([accuracy_row], columns=accuracy_df.columns)
    accuracy_df = pd.concat([accuracy_df, accuracy_row_df])

    accuracy_df.to_csv('..\\Logs\\inference_metrics_fin.csv')


def save_model(edges, data, n, use_black, use_white, use_fixed_edges, use_corr, use_rand):
    """
    This method writes a new file, which describes the computed Bayesian Network, to the BayesianModels directory.

    """
    edge_filename, node_filename = compute_filenames(n, use_white, use_black, use_fixed_edges, use_corr, use_rand)

    edges_df = pd.DataFrame(data=list(edges), columns=['from', 'to'])
    edges_df.replace('-', '.', regex=True, inplace=True)
    edges_df.to_csv(edge_filename)

    nodes = list(set(itertools.chain.from_iterable(edges)))
    data[nodes].to_csv(node_filename)


def remove_duplicates_from_list_of_tuples(lst):
    seen = set()
    result = []

    for tpl in lst:
        if tpl not in seen:
            seen.add(tpl)
            result.append(tpl)

    return result


def determine_selection_method(use_corr, use_rand):
    method = 'OnPLs'
    if use_corr:
        method = 'Correlation'
    elif use_rand:
        method = 'Random'

    return method


def determine_config(use_black, use_white):
    config = 'No Config'
    if use_black and use_white:
        config = 'Config(wb)'
    elif use_black:
        config = 'Config(b)'
    elif use_white:
        config = 'Config(w)'

    return config


def method_name_pipeline(use_corr, use_rand, use_black, use_white, use_pc):
    selection_method = determine_selection_method(use_corr, use_rand)
    config = determine_config(use_black, use_white)
    structure_method = 'PC' if use_pc else 'HillClimb'

    return selection_method, config, structure_method


def building_pipeline(data, use_3tiers, learn_network, log_matching, use_pc, top_loadings, nodes_per_block, use_white,
                      use_black, use_fixed_edges, show_network_physics):
    """
    This method represents the steps taken to build the Bayesian Network.
    The meaning of all of the parameters names are explained in the methods of each method used in the pipeline.
    """
    discrete_data = discretize_data(data, use_3tiers, learn_network)
    if log_matching and not use_pc:
        model = structure_learning(data, top_loadings, nodes_per_block, use_pc, use_black, use_white, use_fixed_edges)
    else:
        model = structure_learning(discrete_data, top_loadings, nodes_per_block, use_pc, use_black, use_white,
                                   use_fixed_edges)

    if show_network_physics:
        visualize_network(model, data)
    return model, discrete_data


def matching_pipeline(model, top_loadings, genes_each_miRNA, log_matching, nodes_per_block, selection_method,
                      config, structure_method, show_network):
    """
    This method represents the steps taken to find,determine and log the matches between the edges of a computed BN, and of 'real' edges, found by biological experiments.
    The meaning of all of the parameters names' are explained in the methods used in the pipeline.
    """
    microRNA_TopGenes = filter_mRNA(top_loadings, genes_each_miRNA)
    df_id = get_string_identifiers(model, microRNA_TopGenes)
    interactions = request_interactions(df_id)
    interactions = map_interactions(interactions, df_id)

    determine_match_count(process_interactions(interactions), model, log_matching, nodes_per_block,
                          selection_method,
                          config, structure_method)

    if show_network:
        visualize_string_network(interactions, 'Interactions - Real')
        visualize_string_network(pd.DataFrame(list(model.edges), columns=['from', 'to']), 'Interactions - Computed')


def inference_pipeline(discrete_data, model, nodes_per_block, selection_method, config, log_accuracy):
    """
    This method represents the steps taken to: keep only the connected structure of the network (where 'Label' is),compute the CPDs of the nodes,classify the Label, log the classification metrics
    The meaning of all of the parameters names' are explained in the methods used in the pipeline.
    """

    connected_data, connected_edges = keep_needed_edges(discrete_data, model.edges)
    metrics = fit_network(connected_data, connected_edges)

    if log_accuracy:
        log_network_accuracy(metrics, model, nodes_per_block, selection_method, config)


def pipeline(nodes_per_block=10, use_corr=False, use_rand=False, genes_each_miRNA=10, use_black=True, use_white=True,
             use_pc=False,
             use_fixed_edges=False,
             use_3tiers=False,
             learn_network=True, compute_matches=False, save_network=True, show_network=True,
             show_network_physics=False, log_matching=False,
             log_accuracy=False):
    """
    This method is the main pipeline for the entire BN construction,matching and classifying process.
    The meaning of all of the parameters names' are explained in the methods of each pipeline, namely : building,inference,matching pipeline.

    """
    if log_accuracy: learn_network = True
    if log_matching: compute_matches = True

    data, top_loadings = setup_data(nodes_per_block, use_corr, use_rand, learn_network)
    selection_method, config, structure_method = method_name_pipeline(use_corr, use_rand, use_black, use_white, use_pc)

    model, discrete_data = building_pipeline(data, use_3tiers, learn_network, log_matching, use_pc, top_loadings,
                                             nodes_per_block, use_white,
                                             use_black, use_fixed_edges, show_network_physics)

    if learn_network:
        inference_pipeline(discrete_data, model, nodes_per_block, selection_method, config, log_accuracy)

    if compute_matches:
        matching_pipeline(model, top_loadings, genes_each_miRNA, log_matching, nodes_per_block, selection_method,
                          config, structure_method, show_network)

    if save_network:
        save_model(model.edges, data, nodes_per_block, use_black, use_white, use_fixed_edges, use_corr, use_rand)

    return model


if __name__ == '__main__':
    pipeline(
        # configuration: building the network
        nodes_per_block=5
        , use_corr=False
        , use_rand=False
        , use_black=False
        , use_white=False
        , use_pc=False
        , use_3tiers=True
        , use_fixed_edges=False

        # configuration: inference or matching
        , compute_matches=False
        , log_matching=False
        , log_accuracy=False
        , learn_network=True

        # configuration: auxiliary methods
        , show_network=True
        , show_network_physics=False
        , save_network=False
    )
