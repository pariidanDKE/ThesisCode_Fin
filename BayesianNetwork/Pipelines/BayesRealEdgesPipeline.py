import pandas as pd
import requests
import itertools
import numpy as np
from pgmpy.estimators import K2Score
from pgmpy.estimators import HillClimbSearch

from BayesianNetwork.Pipelines.DicretizeData import discretize_data
from BayesianNetwork.Pipelines.BayesPipeline import fit_network, keep_needed_edges, \
    remove_duplicates_from_list_of_tuples


'''
The pipeline in this file is not used in the thesis, as there was no room for it.

However, the file contains the pipeline used to computed a network from real edges, and test its inference capabilities ( some errors may occur, as its not been updated in a while)
'''


def setup_data(n=10, use_corr=False):
    df_mir = pd.read_csv('../Data/mir_data_labelled.csv', index_col=0)
    df_pr = pd.read_csv('../Data/pr_data_labelled.csv', index_col=0)
    df_mrna = pd.read_csv('../Data/mrna_data_labelled.csv', index_col=0)
    df_pheno = pd.read_csv('../Data/phenotype.csv')

    if use_corr:
        top_loadings = pd.read_csv('../Data/BayesVariables_Allblocks_CorrelationOnly.csv', index_col=0)
    else:
        top_loadings = pd.read_csv('../Data/BayesVariables_Allblocks_OnlyPC1.csv', index_col=0)

    # n = 20
    df_mir_top = df_mir[top_loadings['microRNA'].head(n)]
    df_pr_top = df_pr[top_loadings['Protein'].head(n)]
    df_mrna_top = df_mrna[top_loadings['mRNA'].head(n)]

    data = df_mir_top.join(df_mrna_top).join(df_pr_top)
    data['Label'] = df_mir['Braak']
    data['Label'] = data['Label'].replace('Case', 1)
    data['Label'] = data['Label'].replace('Control', 0)

    # df_mir_top['Braak'] = df_mir['Braak']
    # df_pr_top['Braak'] = df_pr['Braak']
    # df_mrna_top['Braak'] = df_mrna['Braak']

    return data, top_loadings, df_mir_top, df_pr_top, df_mrna_top


def filter_mRNA(top_loadings, df_mir, top_gene_count=5):
    mir_interactions_df = pd.read_csv('../Data/mir_interactions_onlyhsa.csv')
    # mir_interactions_df = mir_interactions_df[mir_interactions_df['miRNA'].str.contains("hsa")]
    # mir_interactions_df.drop_duplicates(inplace=True)

    top_loadings = top_loadings.loc[top_loadings['microRNA'].isin(df_mir.columns)]
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


def mir_id(microRNA_TopGenes, df_mir_id):
    df_mir_id = microRNA_TopGenes.rename(columns={"microRNA": "MyDataName", "Target Gene": "preferredName"})
    df_mir_id = df_mir_id[df_mir_id['preferredName'].apply(lambda x: isinstance(x, str))]
    return df_mir_id[['MyDataName', 'preferredName']]


def mrna_id(top_loadings, df_mrna_id):
    mRNA_list = df_mrna_id.columns.tolist()

    middle_point = int(len(mRNA_list) / 2)

    mRNA_list1 = mRNA_list[:middle_point]
    mRNA_list2 = mRNA_list[middle_point:]

    mRNAs1 = '%0d'.join(mRNA_list1)

    url = 'https://string-db.org//api/tsv/get_string_ids?identifiers=' + mRNAs1 + '&species=9606'
    r = requests.get(url)
    lines = r.text.split('\n')
    data1 = [l.split('\t') for l in lines]

    mRNAs2 = '%0d'.join(mRNA_list2)

    url = 'https://string-db.org//api/tsv/get_string_ids?identifiers=' + mRNAs2 + '&species=9606'
    r = requests.get(url)
    lines = r.text.split('\n')
    data2 = [l.split('\t') for l in lines]

    df_mrnaid_1 = pd.DataFrame(data1[1:-1], columns=data1[0])
    df_mrnaid_2 = pd.DataFrame(data2[1:-1], columns=data2[0])
    df_mrna_id = pd.concat([df_mrnaid_1, df_mrnaid_2])

    found_mRNAs = [mRNA_list[int(q_idx)] for q_idx in list(df_mrna_id.queryIndex.values)]
    df_mrna_id['MyDataName'] = found_mRNAs

    return df_mrna_id[['MyDataName', 'preferredName']]


def pr_id(top_loadings, df_pr_id):
    # removing hyphens from proteins

    #  for idx, row in top_loadings.iterrows():
    #    top_loadings.loc[idx, 'Protein'] = row['Protein'].partition("-")[0]

    protein_list = df_pr_id.columns.tolist()  # top_loadings['Protein'].to_list()
    for i in range(0, len(protein_list)):
        protein_list[i] = protein_list[i].partition("-")[0]

    proteins = '%0d'.join(protein_list)

    url = 'https://string-db.org//api/tsv/get_string_ids?identifiers=' + proteins + '&species=9606'
    r = requests.get(url)
    lines = r.text.split('\n')
    data = [l.split('\t') for l in lines]

    df_pr_id = pd.DataFrame(data[1:-1], columns=data[0])
    found_proteins = [protein_list[int(q_idx)] for q_idx in list(df_pr_id.queryIndex.values)]
    df_pr_id['MyDataName'] = found_proteins

    return df_pr_id[['MyDataName', 'preferredName']]


def total_id(df_pr_id, df_mrna_id, df_mir_id):
    df_id = df_pr_id.append(df_mrna_id, ignore_index=True).append(df_mir_id)
    df_id.reset_index(inplace=True)
    df_id.drop(['index'], inplace=True, axis=1)

    return df_id


def request_interactions(df_id):
    identifiers = '%0d'.join(list(df_id['preferredName'].unique()[:500]))
    url = 'https://string-db.org/api/tsv/network?identifiers=' + identifiers + '&species=9606'
    r = requests.get(url)

    lines = r.text.split('\n')  # pull the text from the response object and split based on new lines
    data = [l.split('\t') for l in lines]  # split each line into its components based on tabs
    # convert to dataframe using the first row as the column names; drop empty, final row
    df = pd.DataFrame(data[1:-1], columns=data[0])

    identifiers2 = '%0d'.join(list(df_id['preferredName'].unique()[:500]))
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
    interactions_mapped = pd.DataFrame(columns=['from', 'to', 'score'])

    for i in range(1, len(interactions)):

        mapped_A = df_id.loc[df_id['preferredName'] == interactions.iloc[i]['preferredName_A']]
        mapped_B = df_id.loc[df_id['preferredName'] == interactions.iloc[i]['preferredName_B']]

        pairwise_all = list(
            itertools.product((list(mapped_A['MyDataName'].values)), (list(mapped_B['MyDataName'].values))))
        pairwise_df = pd.DataFrame(pairwise_all, columns=['from', 'to', ])
        pairwise_df['score'] = interactions.iloc[i]['score']

        if len(mapped_A) > 0:
            interactions_mapped = pd.concat([interactions_mapped, pairwise_df])

    return interactions_mapped.sort_values(by='score', ascending=False)


def get_intersect(df, interactions, n):
    df_top_a = df[df.columns.intersection(interactions['from'].head(n))]
    df_top_b = df[df.columns.intersection(interactions['to'].head(n))]

    df_top = df_top_a.merge(df_top_b, left_index=True, right_index=True, suffixes=('', '_delme'))
    df_top = df_top[[c for c in df_top.columns if not c.endswith('_delme')]]

    # df_top = df_top.T.drop_duplicates().T

    return df_top


def rename_cols(df_pr):
    # print(df_pr.columns)
    for col in df_pr.columns:
        new_col = col.partition("-")[0]
        if new_col in df_pr.columns and new_col != col:
            print('Found duplicate column!', new_col)
            df_pr.drop([new_col], axis=1, inplace=True)
        df_pr.rename(columns={col: new_col}, inplace=True)

    return df_pr


def setup_nodes(interactions, df_mir, df_pr, df_mrna, data1, edges_count):
    df_mir_top = get_intersect(df_mir, interactions, edges_count)
    df_pr_top = get_intersect(df_pr, interactions, edges_count)
    df_mrna_top = get_intersect(df_mrna, interactions, edges_count)

    data = df_pr_top.join(df_mrna_top)
    data = data.join(df_mir_top)
    data['Label'] = data1['Label'].values

    return data


def train_network(data, interactions, edges_count):
    edges = [tuple([i[0], i[1]]) for i in np.array(interactions.head(edges_count))]  # %%
    edges = remove_duplicates_from_list_of_tuples(edges)  # [t for t in (set(tuple(i) for i in edges))]

    nodes = list(data.columns)
    nodes.remove('Label')

    white_list = edges
    white_list.extend(list(itertools.product(nodes, ['Label'])))

    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=K2Score(data), white_list=white_list)

    return best_model, nodes, edges


def save_model(edges, data, n, edges_count, genes_each_mir, use_corr, use_rand):
    edge_filename, node_filename = compute_filenames(n, edges_count, genes_each_mir, use_corr, use_rand)

    edges_df = pd.DataFrame(data=list(edges), columns=['from', 'to'])
    edges_df.replace('-', '.', regex=True, inplace=True)
    edges_df.to_csv(edge_filename)

    nodes = list(set(itertools.chain.from_iterable(edges)))
    data[nodes].to_csv(node_filename)


def compute_filenames(n, edges_count, genes_each_mir, use_corr, use_rand):
    edge_filename = '..\\BayesianModels\\real_edges' + str(n) + '_n_' + str(edges_count) + 'e_' + str(
        genes_each_mir) + 'gc'
    node_filename = '..\\BayesianModels\\real_nodes' + str(n) + '_n_' + str(edges_count) + 'e_' + str(
        genes_each_mir) + 'gc'

    if use_corr:
        edge_filename += '_corr_'
        node_filename += '_corr_'

    if use_rand:
        edge_filename += '_rand_'
        node_filename += '_rand_'

    edge_filename += '.csv'
    node_filename += '.csv'

    return edge_filename, node_filename


def pipeline(nodes_per_block, genes_each_miRNA, edges_count, save_network=False, use3_tiers=False, use_corr=False,
             use_rand=False):
    data, top_loadings, df_mir, df_pr, df_mrna = setup_data(nodes_per_block, use_corr)
    microRNA_TopGenes = filter_mRNA(top_loadings, df_mir, genes_each_miRNA)

    df_mrna_id = mrna_id(top_loadings, df_mrna)
    df_pr_id = pr_id(top_loadings, df_pr)
    df_mir_id = mir_id(microRNA_TopGenes, df_mir)

    df_id = total_id(df_pr_id, df_mrna_id, df_mir_id)
    interactions = request_interactions(df_id)
    interactions = map_interactions(interactions, df_id)

    network_data = setup_nodes(interactions, df_mir, rename_cols(df_pr), df_mrna, data, edges_count)
    discrete_data = discretize_data(network_data, use3_tiers, False)
    model, nodes, edges_space = train_network(discrete_data, interactions, edges_count)

    # determine_match_count(edges_space, model.copy())

    connected_data, connected_edges = keep_needed_edges(discrete_data, model.edges)
    fit_network(connected_data, connected_edges)

    if save_network:
        save_model(model.edges, network_data, nodes_per_block, edges_count, genes_each_miRNA, use_corr, use_rand)


if __name__ == '__main__':
    pipeline(nodes_per_block=10, genes_each_miRNA=20, edges_count=3000, save_network=False, use3_tiers=True,
             use_corr=False)

## Q9NQC3 for 0.483
## P00505 for 0.7666666666667
## hsa-miR-26a-5p 0.816666668
## 'hsa-miR-326' 0.816666
## P62333 0.7833333333333333
## ENSG00000165092 0.783333333333333
