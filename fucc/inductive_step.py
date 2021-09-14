from multiprocessing import Pool
import pandas as pd
from functools import partial
import numpy as np
from tqdm import tqdm


def inductive_pooling(df, embeddings, G, workers, gamma=1000, dict_node=None, average_embedding=True):
    


    if average_embedding:
        avg_emb = embeddings.mean().values
    else:
        avg_emb = None
    
    #if __name__ == '__main__':
    with Pool(workers) as p:
        r = p.map(partial(inductive_pooling_chunk, embeddings=embeddings, G=G, average_embedding=avg_emb), np.array_split(df, workers))

    return r
	
def inductive_pooling_chunk(df, embeddings, G, gamma=1000, average_embedding=None):
	
    #Create a container for the new embeddings
    new_embeddings = dict()

    for transaction, transaction_row in tqdm(df.iterrows(), total=df.shape[0]):

        cardholder = transaction_row.CARD_PAN_ID
        merchant = transaction_row.TERM_MIDUID


        mutual = False

        if G.has_node(cardholder) & G.has_node(merchant):
            mutual_neighbors = list(set(G.neighbors(cardholder)).intersection(set(G.neighbors(merchant))))
            mutual_neighbors.sort()
            if (len(mutual_neighbors) > 0): 
                mutual = True
                # Use dataframe with TX_ID on index (to speed up retrieval of transaction rows)
                embeddings_mutual_neighbors = embeddings.loc[mutual_neighbors]
                # most recent transaction
                most_recent_embedding_mutual_neighbor = embeddings_mutual_neighbors.iloc[-1]

                new_embeddings[transaction] = most_recent_embedding_mutual_neighbor
                
                        
        if G.has_node(cardholder) & (not mutual):

            cardholder_neighbors = list(G.neighbors(cardholder))
            
            pooled_embedding = get_pooled_embedding(cardholder_neighbors, embeddings, gamma)
            
            new_embeddings[transaction] = pooled_embedding
            
        elif G.has_node(merchant) & (not mutual):

            merchant_neighbors = list(G.neighbors(merchant))

            pooled_embedding = get_pooled_embedding(merchant_neighbors, embeddings, gamma)

            new_embeddings[transaction] = pooled_embedding
            
            
        elif (not mutual):
            new_embeddings[transaction] = average_embedding
                    

    return new_embeddings
								
def get_pooled_embedding(neighbors, embeddings, gamma):
	
	embeddings_to_pool = embeddings.loc[neighbors]
	most_recent_embeddings_to_pool = embeddings_to_pool.iloc[-min(gamma, embeddings_to_pool.shape[0]):]
	
	pooled_embedding = pd.DataFrame(most_recent_embeddings_to_pool.mean()).transpose().values[0]
	
	return pooled_embedding
	
