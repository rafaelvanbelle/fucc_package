from multiprocessing import Pool
import pandas as pd
from functools import partial
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import random

def calculate_average_embedding(df_before_TX_index, embeddings, dict_node=None):
    """[summary]

    Args:
        df_before_TX_index ([type]): [description]
        embeddings ([type]): [description]
        dict_node ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    
    # Get an average transaction embedding (for the worst case scenario)
    df_embeddings = dict()
    for i, row in df_before_TX_index.iterrows():
        if dict_node:
            df_embeddings[str(i)] = embeddings.loc[str(dict_node[str(i)])]
        else:
            df_embeddings[str(i)] = embeddings.loc[str(i)]

    df_embeddings = pd.DataFrame().from_dict(df_embeddings, orient='index')
    average_embedding = df_embeddings.mean()
    return average_embedding



def inductive_nn(df_today, df_before, embeddings, G, workers, transaction_node_features, dict_node=None, average_embedding=True):
    """[summary]

    Args:
        df_today ([type]): [description]
        df_before ([type]): [description]
        embeddings ([type]): [description]
        G ([type]): [description]
        workers ([type]): [description]
        transaction_node_features ([type]): [description]
        dict_node ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """


    df_before_TX_index = df_before.set_index('TX_ID')

    if average_embedding:
        avg_emb = calculate_average_embedding(df_before_TX_index, embeddings, dict_node)
    else:
        avg_emb = None
    
    #if __name__ == '__main__':
    with Pool(workers) as p:
        r = p.map(partial(inductive_nn_chunk, df_before_TX_index=df_before_TX_index, embeddings=embeddings, G=G, dict_node=dict_node, average_embedding=avg_emb, transaction_node_features=transaction_node_features), np.array_split(df_today, workers))

    return r

def inductive_nn_chunk(df_today, df_before_TX_index ,embeddings, G, average_embedding, transaction_node_features, dict_node=None):
    """[summary]

    Args:
        df_today ([type]): [description]
        df_before_TX_index ([type]): [description]
        embeddings ([type]): [description]
        G ([type]): [description]
        average_embedding ([type]): [description]
        transaction_node_features ([type]): [description]
        dict_node ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    #Create a container for the new embeddings
    new_embeddings = dict()
    
    #Keep track of the statistics
    stats = {'both':0, 'cardholder':0, 'merchant':0, 'none':0}
    setting_dict = {}

    for transaction, transaction_row in tqdm(df_today.iterrows(), total=df_today.shape[0]):
        mms = MinMaxScaler()
        cardholder = transaction_row.CARD_PAN_ID
        merchant = transaction_row.TERM_MIDUID
        
        nearest_neighbor_cardholder = None
        nearest_neighbor_merchant = None

        embedding_nn_cardholder = None
        embedding_nn_merchant = None

        if G.has_node(cardholder):
            try: 
                
                try:
                    neighbors_cardholder = random.sample(list(G.neighbors(cardholder)), 10)
                except ValueError:
                    neighbors_cardholder = G.neighbors(cardholder)
                    
                # Use dataframe with TX_ID on index (to speed up retrieval of transaction rows)
                df_cardholder = df_before_TX_index.loc[neighbors_cardholder]
                # Append current transaction 
                df_cardholder = df_cardholder.append(transaction_row)
                # Normalize rows (min_max_scaler)
                df_cardholder_normalized = mms.fit_transform(df_cardholder.loc[:, transaction_node_features])

                dist = np.linalg.norm(df_cardholder_normalized[:-1] - df_cardholder_normalized[-1], axis=1)
                nearest_neighbor_cardholder = df_cardholder.iloc[np.argmin(dist)].name
                
                # If there is a dict_node, we need to translate the node into the original label
                if dict_node:
                    embedding_nn_cardholder = embeddings.loc[str(dict_node[str(nearest_neighbor_cardholder)])]
                else:
                    embedding_nn_cardholder = embeddings.loc[str(nearest_neighbor_cardholder)]
            except:
                embedding_nn_cardholder = None
                nearest_neighbor_cardholder = None

        if G.has_node(merchant):
            try: 

                try:
                    neighbors_merchant = random.sample(list(G.neighbors(merchant)), 10)
                except ValueError:
                    neighbors_merchant = G.neighbors(merchant)
                # Use dataframe with TX_ID on index (to speed up retrieval of transaction rows)
                df_merchant = df_before_TX_index.loc[neighbors_merchant]
                # Append current transaction 
                df_merchant = df_merchant.append(transaction_row)
                # Normalize rows (min_max_scaler)
                df_merchant_normalized = mms.fit_transform(df_merchant.loc[:, transaction_node_features])

                dist = np.linalg.norm(df_merchant_normalized[:-1] - df_merchant_normalized[-1], axis=1)
                nearest_neighbor_merchant = df_merchant.iloc[np.argmin(dist)].name
                if dict_node:
                    embedding_nn_merchant = embeddings.loc[str(dict_node[str(nearest_neighbor_merchant)])]
                else:
                    embedding_nn_merchant = embeddings.loc[str(nearest_neighbor_merchant)]
            except:
                embedding_nn_merchant = None
                nearest_neighbor_merchant = None

        if (nearest_neighbor_cardholder != None) & (nearest_neighbor_merchant != None):
            new_embeddings[transaction] = (embedding_nn_cardholder + embedding_nn_merchant)/2
            stats['both'] += 1
            setting_dict[transaction] = 'both'

        elif nearest_neighbor_cardholder: 
            new_embeddings[transaction] = embedding_nn_cardholder
            stats['cardholder'] += 1
            setting_dict[transaction] = 'cardholder'

        elif nearest_neighbor_merchant:
            new_embeddings[transaction] = embedding_nn_merchant
            stats['merchant'] += 1
            setting_dict[transaction] = 'merchant'

        else:
            new_embeddings[transaction] = average_embedding
            stats['none'] += 1
            setting_dict[transaction] = 'none'
            
    return new_embeddings, stats, setting_dict



def inductive_nn_v2(df_today, df_before, embeddings, G, workers, transaction_node_features, dict_node=None, average_embedding=True):
    """[summary]

    Args:
        df_today ([type]): [description]
        df_before ([type]): [description]
        embeddings ([type]): [description]
        G ([type]): [description]
        workers ([type]): [description]
        transaction_node_features ([type]): [description]
        dict_node ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """


    df_before_TX_index = df_before.set_index('TX_ID')

    if average_embedding:
        avg_emb = calculate_average_embedding(df_before_TX_index, embeddings, dict_node)
    else:
        avg_emb = None
    
    #if __name__ == '__main__':
    with Pool(workers) as p:
        r = p.map(partial(inductive_chunk, df_before_TX_index=df_before_TX_index, embeddings=embeddings, G=G, dict_node=dict_node, average_embedding=avg_emb, transaction_node_features=transaction_node_features), np.array_split(df_today, workers))

    return r



def inductive_chunk(df_today, df_before_TX_index ,embeddings, G, average_embedding, transaction_node_features, dict_node=None):
    """[summary]

    Args:
        df_today ([type]): [description]
        df_before_TX_index ([type]): [description]
        embeddings ([type]): [description]
        G ([type]): [description]
        average_embedding ([type]): [description]
        transaction_node_features ([type]): [description]
        dict_node ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    #Create a container for the new embeddings
    new_embeddings = dict()
    #If both cardholder and merchant have been seen before but no transaction was made, we will save both cardholder and merchant embedding
    second_embeddings = dict()

    #Keep track of the statistics
    stats = {'both':0, 'cardholder':0, 'merchant':0, 'none':0, 'most_recent':0}
    setting_dict = {}

    for transaction, transaction_row in tqdm(df_today.iterrows(), total=df_today.shape[0]):
        mms = MinMaxScaler()
        cardholder = transaction_row.CARD_PAN_ID
        merchant = transaction_row.TERM_MIDUID
        
        
        most_recent_transaction = None
        embedding_most_recent_transaction = None

        if G.has_node(cardholder) & G.has_node(merchant):
            mutual_neighbors = list(set(G.neighbors(cardholder)).intersection(set(G.neighbors(merchant))))
            if len(mutual_neighbors) > 0:
                # Use dataframe with TX_ID on index (to speed up retrieval of transaction rows)
                df_mutual = df_before_TX_index.loc[mutual_neighbors]
                # Sort rows on TX_DATETIME
                df_mutual = df_mutual.sort_values(by='TX_DATETIME', ascending=True)
                # most recent transaction
                most_recent_transaction = df_mutual.iloc[-1].name

            # If there is a dict_node, we need to translate the node into the original label
                if dict_node:
                    embedding_most_recent_transaction = embeddings.loc[str(dict_node[str(most_recent_transaction)])]
                else:
                    embedding_most_recent_transaction = embeddings.loc[str(most_recent_transaction)]

                new_embeddings[transaction] = embedding_most_recent_transaction
                stats['most_recent'] += 1
                setting_dict[transaction] = 'most_recent'

            else:
                # Set this value to avoid next if statement from executing
                most_recent_transaction = 1

                # get most recent cardholder tx
                cardholder_neighbors = list(G.neighbors(cardholder))
                df_cardholder_neighbors = df_before_TX_index.loc[cardholder_neighbors]
                # Sort rows on TX_DATETIME
                df_cardholder_neighbors = df_cardholder_neighbors.sort_values(by='TX_DATETIME', ascending=True)
                # most recent transaction
                most_recent_transaction_cardholder = df_cardholder_neighbors.iloc[-1].name

                # get cardholder embedding
                if dict_node:
                    embedding_cardholder = embeddings.loc[str(dict_node[str(most_recent_transaction_cardholder)])]
                else:
                    embedding_cardholder = embeddings.loc[str(most_recent_transaction_cardholder)]

                
                # get most recent merchant tx
                merchant_neighbors = list(G.neighbors(merchant))
                df_merchant_neighbors = df_before_TX_index.loc[merchant_neighbors]
                # Sort rows on TX_DATETIME
                df_merchant_neighbors = df_merchant_neighbors.sort_values(by='TX_DATETIME', ascending=True)
                # most recent transaction
                most_recent_transaction_merchant = df_merchant_neighbors.iloc[-1].name

                # get merchant embedding 
                if dict_node:
                    embedding_merchant = embeddings.loc[str(dict_node[str(most_recent_transaction_merchant)])]
                else:
                    embedding_merchant = embeddings.loc[str(most_recent_transaction_merchant)]

                new_embeddings[transaction] = embedding_cardholder
                second_embeddings[transaction] = embedding_merchant
                
                stats['both'] += 1
                setting_dict[transaction] = 'both'

        if most_recent_transaction == None:
            if G.has_node(cardholder):

                cardholder_neighbors = list(G.neighbors(cardholder))
                df_cardholder_neighbors = df_before_TX_index.loc[cardholder_neighbors]
                # Sort rows on TX_DATETIME
                df_cardholder_neighbors = df_cardholder_neighbors.sort_values(by='TX_DATETIME', ascending=True)
                # most recent transaction
                most_recent_transaction_cardholder = df_cardholder_neighbors.iloc[-1].name



                if dict_node:
                    embedding_cardholder = embeddings.loc[str(dict_node[str(most_recent_transaction_cardholder)])]
                else:
                    embedding_cardholder = embeddings.loc[str(most_recent_transaction_cardholder)]

                new_embeddings[transaction] = embedding_cardholder
                stats['cardholder'] += 1
                setting_dict[transaction] = 'cardholder'

            elif G.has_node(merchant):

                merchant_neighbors = list(G.neighbors(merchant))
                df_merchant_neighbors = df_before_TX_index.loc[merchant_neighbors]
                # Sort rows on TX_DATETIME
                df_merchant_neighbors = df_merchant_neighbors.sort_values(by='TX_DATETIME', ascending=True)
                # most recent transaction
                most_recent_transaction_merchant = df_merchant_neighbors.iloc[-1].name



                if dict_node:
                    embedding_merchant = embeddings.loc[str(dict_node[str(most_recent_transaction_merchant)])]
                else:
                    embedding_merchant = embeddings.loc[str(most_recent_transaction_merchant)]

                new_embeddings[transaction] = embedding_merchant
                stats['merchant'] += 1
                setting_dict[transaction] = 'merchant'
            
            else:
                new_embeddings[transaction] = average_embedding
                stats['none'] += 1
                setting_dict[transaction] = 'none'

        
            
    return new_embeddings, stats, setting_dict, second_embeddings
	
def inductive_pooling(df, embeddings, G, workers, gamma=1000, dict_node=None, average_embedding=True):
	
	#Create a container for the new embeddings
	new_embeddings = dict()
	
	#Keep track of the statistics
	stats = {'both':0, 'cardholder':0, 'merchant':0, 'none':0, 'most_recent':0}
	setting_dict = {}
	
	for transaction, transaction_row in tqdm(df.iterrows(), total=df.shape[0]):
		
		cardholder = transaction_row.CARD_PAN_ID
		merchant = transaction_row.TERM_MIDUID
		
		
		most_recent_transaction = None
		embedding_most_recent_transaction = None
		
		if G.has_node(cardholder) & G.has_node(merchant):
			mutual_neighbors = list(set(G.neighbors(cardholder)).intersection(set(G.neighbors(merchant))))
			if len(mutual_neighbors) > 0:
				# Use dataframe with TX_ID on index (to speed up retrieval of transaction rows)
				embeddings_mutual_neighbors = embeddings.loc[mutual_neighbors] 
				
				# most recent transaction
				most_recent_embedding_mutual_neighbor = embeddings_mutual_neighbors.iloc[-1]
				
				new_embeddings[transaction] = most_recent_embedding_mutual_neighbor
				stats['most_recent'] += 1
				setting_dict[transaction] = 'most_recent'
		
		elif G.has_node(cardholder):
		
			cardholder_neighbors = list(G.neighbors(cardholder))
			
			pooled_embedding = get_pooled_embedding(cardholder_neighbors, embeddings, gamma)
			
			new_embeddings[transaction] = pooled_embedding
			stats['cardholder'] += 1
			setting_dict[transaction] = 'cardholder'
		
		elif G.has_node(merchant):
		
			merchant_neighbors = list(G.neighbors(merchant))
			
			pooled_embedding = get_pooled_embedding(merchant_neighbors, embeddings, gamma)
			
			new_embeddings[transaction] = pooled_embedding
			stats['merchant'] += 1
			setting_dict[transaction] = 'merchant'
			
		else:
			new_embeddings[transaction] = average_embedding
			stats['none'] += 1
			setting_dict[transaction] = 'none'
			
			
			
	
			return new_embeddings, stats, setting_dict
			
					
def get_pooled_embedding(neighbors, embeddings, gamma):
	
	embeddings_to_pool = embeddings.loc[neighbors]
	most_recent_embeddings_to_pool = embeddings_to_pool.iloc[-min(gamma, embeddings_to_pool.shape[0]):]
	
	pooled_embedding = pd.DataFrame(most_recent_embeddings_to_pool.mean()).transpose()
	
	return pooled_embedding
	
