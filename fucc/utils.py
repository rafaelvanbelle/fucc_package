
import networkx as nx
import logging
from multiprocessing import Pool
import os
import pickle
import pandas as pd
from tqdm import tqdm

# Create a custom logger
logger = logging.getLogger("utils")


def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])

def export_network(network, output_filepath):
    logger.info("Converting node labels to integers")
    G = nx.convert_node_labels_to_integers(network, first_label=0, ordering='default', label_attribute='label')
    logger.info("Exporting edgelist")
    logger.info(output_filepath)
    nx.write_edgelist(G, output_filepath, comments='#', delimiter=' ', data=False, encoding='utf-8')

    node_dict = dict(G.nodes(data='label'))
    dict_node = invert_dict(node_dict)

    return dict_node

def get_filename(filename_elements = []):
    filename = '_'.join([str(x) for x in filename_elements])
    return filename

def export_suspicion_scores(suspicion_scores, G, filename_elements = [], output_filepath=''):
    
    filename = get_filename(filename_elements=filename_elements)
    
    # Export raw suspicion scores
    f = open(os.path.join(output_filepath, filename + '_pagerank_suspicion_scores.pkl'), "wb")
    pickle.dump(suspicion_scores,f)
    f.close()

    # Export suspicion scores in Pandas DataFrame CSV
    to_save = pd.DataFrame.from_dict(suspicion_scores, orient='index')
    name = os.path.join(output_filepath, filename + '_pagerank_suspicion_scores.csv')
    to_save.to_csv(name)

    # Export Graph
    nx.write_gpickle(G, path = os.path.join(output_filepath, filename + '_graph.pkl'))



def import_suspicion_scores(filename_elements = [], input_filepath=''):
    
    filename = get_filename(filename_elements=filename_elements)
    
    # get the suspicion scores
    # get the graph
    filename_graph = os.path.join(input_filepath, filename + '_graph.pkl')
    filename_scores = os.path.join(input_filepath, filename + '_pagerank_suspicion_scores.pkl')

    G = nx.read_gpickle(filename_graph)
    with open(filename_scores, 'rb') as handle:
        suspicion_scores = pickle.load(handle)
        
    return suspicion_scores, G

def multiprocessing(function, chunks, workers=8):
        
    # multiprocessing
    with Pool(workers) as p:
        r = list(tqdm(p.imap(function, chunks), total=len(chunks)))

    p.close()
    concatenated_result = pd.concat(r)
    return concatenated_result
