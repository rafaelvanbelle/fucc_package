import numpy as np
import pandas as pd
import networkx as nx
import dateparser
import math
import pickle
import glob
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import os
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# ~~Parameters~~ #
lambdas = {'ST': 0.03, 'MT': 0.004, 'LT': 0.0001}
data_path = '/Users/Raf/Dropbox/DOC/data/FUCC/'
files = glob.glob(os.path.join(data_path,'pagerank/pagerank_input/data_pagerank_*.csv'))
output_folder = os.path.join(data_path, 'pagerank/pagerank_output/')


def get_pagerank_suspicion_scores(data, 
                                  lambd, 
                                  alpha=0.85, 
                                  n_jobs=5,
                                  personalization_nodes=None,
                                  weighted = True,
                                  t = None):
    """[calculcate pagerank suspicion scores for every rows in data]

    Arguments:
        data {[pandas DataFrame]} -- [description]

    Keyword Arguments:
        t {str} -- [description] (default: {'ST'})
        lambd {float} -- [description] (default: {0.03})
        alpha {float} -- [description] (default: {0.85})
        n_jobs {int} -- [description] (default: {5})
        personalization_nodes {list} -- a list of nodes that will be used in the personalization vector. If None is supplied, all fraudulent nodes
        will be used in the personalization vector. 

    Returns:
        [dict] -- [description]
        [networkx Graph] -- [description]
    """

    logging.info("Building network")


    #weights_data = calculate_edge_weights(data, t=t, lambd=lambd)
    suspicion_scores = dict()
    weights_data_tx_id_key = dict()

    if weighted:
        logging.info('Calculating edge weights for {}'.format(t))
        weights_data = calculate_edge_weights(data, t=t, lambd=lambd)

        weighted_edgelist = list(zip(data.CARD_PAN_ID, data.TX_ID,
                                    [weights_data.get(key) for key in data.index])) + \
                            list((zip(data.TERM_MIDUID, data.TX_ID,
                                    [weights_data.get(key) for key in data.index])))

        
        # determine starting vector
        # starting_vector = ... Has to be time-weighted (See p.155) - update: In the APATE paper this vector e_0 is a random vector. The personalization vector has to be time-weighted, see below.
        # I will not specifiy a non-default e_0 vector (which is referred to as nstart in networkx)

        # personalization
        # Make a new weights dictionary with TX_ID as key
        logging.info('Personalization')
        weights_data_tx_id_key = dict(zip(data.TX_ID, weights_data.values()))

    else:
        # Weighted edgelist with all 1
        weighted_edgelist = list(zip(data.CARD_PAN_ID, data.TX_ID,
                            [1 for key in data.index])) + \
                    list((zip(data.TERM_MIDUID, data.TX_ID,
                            [1 for key in data.index])))

        logging.info('Personalization')
        weights_data_tx_id_key = dict(zip(data.TX_ID, [1 for entry in data.TX_ID]))
    
    logging.info('Building graph {}'.format(t))
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edgelist)

    if personalization_nodes:
        subset = data.loc[personalization_nodes]
        personalization = \
                        {dc['TX_ID']: weights_data_tx_id_key.get(dc['TX_ID']) for dc in subset[subset.TX_FRAUD == True].to_dict(orient='records')}
    else:
        personalization = \
                        {dc['TX_ID']: weights_data_tx_id_key.get(dc['TX_ID']) for dc in data[data.TX_FRAUD == True].to_dict(orient='records')}
    print(G.number_of_nodes())
    print(len(personalization))
    # Note: personalization dictionary is normalized in networkx pagerank implementation

    # run pagerank
    logging.info('Pagerank started')
    suspicion_scores = nx.pagerank(G, alpha=alpha, personalization=personalization, weight='weight')
    logging.info('Pagerank finished')
    
    # as the postprocessing step requires network information, we return the graph G as well.
    return suspicion_scores, G


def calculate_edge_weights(data, t, lambd):
    # determine weights for ST,MT,LT
    logging.info("Calculating some weights")

    current_time = dateparser.parse(data.tail(1).TX_DATETIME.dt.date.values[0].strftime("%d/%m/%Y") + " 23:59:59", settings={'DATE_ORDER': 'DMY'})

    weights_data = dict()
    if t == 'ST':
        logging.info('ok, ST')
        weights_data = (((current_time - data.TX_DATETIME).dt.total_seconds()) / 60).to_dict()
    elif t == 'MT':
        logging.info('ok, MT')
        weights_data = (((current_time - data.TX_DATETIME).dt.total_seconds()) / (60)).to_dict()
    elif t == 'LT':
        logging.info('ok, LT')
        weights_data = (((current_time - data.TX_DATETIME).dt.total_seconds()) / (60)).to_dict()
    else:
        logging.error('The time granularity specified was not recognized.')

    weights_data = {k: math.e ** (-lambd * h) for k, h in weights_data.items()}

    return weights_data


def inductive_step(edges, historical_edges, suspicion_scores, G, t):
    """
    inductive_step is a function which will inductively generate suspicion scores for every edge in edges based on the 
    suspicion_scores of historical observations. 

    param:edges = pandas dataframe containing an edgelist of edges for which you want to calculate a suspicion score
    param:historical_edges = pandas dataframe containing an edgelist of edges observed before the edges in param:edges
    param:suspicion_scores = suspicion_scores of the edges in param:historical_edges
    param:G = networkx graph based on historical_edges
    param:t = time granularity ('ST', 'LT', 'MT')
    """
    logging.info('post processing started')

    # create some new columns to store the suspicion scores
    edges = edges.assign(
        **{'SC_TX_'+t: 0, 'SC_MC_'+t: 0, 'SC_CC_'+t: 0})

    # get suspicion scores
    for i, edge in edges.iterrows():
    
        client = edge.iloc[0]
        merchant = edge.iloc[1]

        indices_client = historical_edges.iloc[:, 0].isin([edge.iloc[0]])
        indices_merchant = historical_edges.iloc[:, 1].isin([edge.iloc[1]])

        client_subset = historical_edges.loc[indices_client]
        merchant_subset = historical_edges.loc[indices_merchant]

        idx1 = client_subset.index
        idx2 = merchant_subset.index
        common_indices = idx1.intersection(idx2)

        if np.any(common_indices):
            
            most_recent_transaction_id = common_indices[-1]
            edges.loc[i, ['SC_TX_' + t]] = \
                suspicion_scores[most_recent_transaction_id]
            edges.loc[i, ['SC_MC_' + t]] = suspicion_scores[merchant]
            edges.loc[i, ['SC_CC_' + t]] = suspicion_scores[client]


        else:
            if np.any(idx1):
                # we know the client exists and had transactions before --> hence get the score
                edges.loc[i, ['SC_CC_' + t]] = suspicion_scores[edge.iloc[0]]
                CCH_score = suspicion_scores[edge.iloc[0]]
                egonet_weights_client = G.edges(edge.iloc[0], data='weight')
                sum_of_egonet_weights_client = np.sum(list(zip(*egonet_weights_client))[2])
                term_client = (1/(sum_of_egonet_weights_client + 1)) * CCH_score
            else:
                # client_score = 0
                edges.loc[i, ['SC_CC_' + t]] = 0
                # term van client valt ook weg
                term_client = 0
            if np.any(idx2):
                # we know the merchant exists and had transactions before --> hence get the score
                edges.loc[i, ['SC_MC_' + t]] = suspicion_scores[edge.iloc[1]]
                MC_score = suspicion_scores[edge.iloc[1]]
                egonet_weights_merchant = G.edges(edge.iloc[1], data='weight')
                sum_of_egonet_weights_merchant = np.sum(list(zip(*egonet_weights_merchant))[2])
                term_merchant = (1/(sum_of_egonet_weights_merchant + 1)) * MC_score
            else:
                # merchant_score = 0
                edges.loc[i, ['SC_MC_' + t]] = 0
                # term van merchant valt weg
                term_merchant = 0
            
            edges.loc[i, ['SC_TX_'+t]] = term_client + term_merchant

    return edges

def postprocessing_historical_edges(historical_edges, suspicion_scores, t):
    """
    The purpose of this method is to gather all suspicion scores of the historical data (non-inductive) into a 
    Pandas Dataframe and save to disk. 

    Args:
        historical_edges ([Pandas DataFrame]): [description]
        suspicion_scores ([type]): [description]
        G ([Networkx Graph]): [description]
        t ([String]): lambda (ST, MT, LT)
    """
    # create some new columns to store the suspicion scores
    historical_edges = historical_edges.assign(
        **{'SC_TX_'+t: 0, 'SC_MC_'+t: 0, 'SC_CC_'+t: 0})

    # get suspicion scores
    for i, edge in historical_edges.iterrows():

        client = edge.iloc[0]
        merchant = edge.iloc[1]
        transaction = i

        historical_edges.loc[i, ['SC_TX_' + t]] = \
                suspicion_scores[transaction]
        historical_edges.loc[i, ['SC_MC_' + t]] = suspicion_scores[merchant]
        historical_edges.loc[i, ['SC_CC_' + t]] = suspicion_scores[client]
    
    return historical_edges

def splitDataFrameIntoSmaller(df, chunkSize = 10000): 
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf


def _pagerank_processing(file):
    
    df = pd.read_csv(file, index_col=0, parse_dates=[7])
    print(df.head())
    historical_data = df.loc[df.TX_DATETIME.dt.date < df.tail(1).TX_DATETIME.dt.date.values[0]]
    data = df.loc[df.TX_DATETIME.dt.date == df.tail(1).TX_DATETIME.dt.date.values[0]]

    for t, lambd in lambdas.items():
        suspicion_scores, G = get_pagerank_suspicion_scores(
                                  historical_data,
                                  t=t,
                                  lambd=lambd,
                                  alpha=0.000085,
                                  n_jobs=1)
        

        f = open(output_folder + Path(file).stem + 'pagerank_suspicion_scores_' + str(t) + '.pkl', "wb")
        pickle.dump(suspicion_scores,f)
        f.close()

        to_save = pd.DataFrame.from_dict(suspicion_scores, orient='index')
        name = output_folder + Path(file).stem + 'pagerank_suspicion_scores_' + str(t) + '.csv'
        to_save.to_csv(name)

        nx.write_gpickle(G, path = output_folder + Path(file).stem + 'graph_'+str(t)+'.pkl')

def _pagerank_postprocessing(file):

    # read the data
    df = pd.read_csv(file, index_col=0, parse_dates=[7])
    
    # split in historical data and current day data
    historical_data = df.loc[df.TX_DATETIME.dt.date < df.tail(1).TX_DATETIME.dt.date.values[0]]
    data = df.loc[df.TX_DATETIME.dt.date == df.tail(1).TX_DATETIME.dt.date.values[0]]

    # We only need these columns from data
    data = data.set_index('TX_ID')
    data = data.loc[:, ['CARD_PAN_ID', 'TERM_MIDUID']]

    # We only need these columns from historical data
    historical_data = historical_data.set_index('TX_ID')
    historical_data = historical_data.loc[:, ['CARD_PAN_ID', 'TERM_MIDUID']]

    for t,lamb in lambdas.items():
        # get the suspicion scores
        # get the graph
        filename_graph = output_folder + Path(file).stem + 'graph_' + str(t) + '.pkl'
        filename_scores = output_folder + Path(file).stem + 'pagerank_suspicion_scores_' + str(t) + '.pkl'
        
        G = nx.read_gpickle(filename_graph)
        with open(filename_scores, 'rb') as handle:
            suspicion_scores = pickle.load(handle)
        
        # split df in smaller chunks
        chunks = splitDataFrameIntoSmaller(data, chunkSize=10000)
        partial_inductive_step = partial(inductive_step, historical_edges=historical_data, suspicion_scores=suspicion_scores, G=G, t=t)

        # multiprocessing
        with Pool(8) as p:
            r = list(tqdm(p.imap(partial_inductive_step, chunks), total=len(chunks)))

        df = pd.concat(r)
        df.to_csv(output_folder + Path(file).stem + '_output_'+str(t)+'.csv')

        #Split historical dataset into smaller chunks
        chunks = splitDataFrameIntoSmaller(historical_data, chunkSize=50000)
        partial_postprocessing_historical_edges = partial(postprocessing_historical_edges, suspicion_scores=suspicion_scores, t=t)
        
        # multiprocessing
        with Pool(8) as p:
            r = list(tqdm(p.imap(partial_postprocessing_historical_edges, chunks), total=len(chunks)))

        df = pd.concat(r)
        df.to_csv(output_folder + Path(file).stem + '_output_historical_edges_'+str(t)+'.csv')

        p.close()



if __name__ == '__main__':
    
    #with Pool(8) as p:
    #   r = list(tqdm(p.imap(_pagerank_processing, files), total=len(files)))


    for fil in tqdm(files, total=len(files)):
        _pagerank_postprocessing(fil)
