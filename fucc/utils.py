
import networkx as nx
import logging

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
