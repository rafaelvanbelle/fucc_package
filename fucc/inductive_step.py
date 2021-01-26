from multiprocessing import Pool
import pandas as pd
from functools import partial



def calculate_average_embedding(df_before_TX_index, embeddings, dict_node=None):
    # Get an average transaction embedding (for the worst case scenario)
    df_embeddings = dict()
    for i, row in df_before_TX_index.iterrows():
        if dict_node:
            df_embeddings[str(i)] = embeddings[str(dict_node[str(i)])]
        else:
            df_embeddings[str(i)] = embeddings[str(i)]

    df_embeddings = pd.DataFrame().from_dict(df_embeddings, orient='index')
    average_embedding = df_embeddings.mean()
    return average_embedding



def inductive_nn(df_today, df_before, embeddings, G, workers, transaction_node_features, dict_node=None):
    
    df_before_TX_index = df_before.set_index('TX_ID')
    
    average_embedding = calculate_average_embedding(df_before_TX_index, embeddings, dict_node)

    
    #if __name__ == '__main__':
    with Pool(workers) as p:
        r = p.map(partial(inductive_nn_chunk, df_before_TX_index=df_before_TX_index, embeddings=embeddings, G=G, dict_node=dict_node, average_embedding=average_embedding, transaction_node_features=transaction_node_features), np.array_split(df_today, workers))

    return r

def inductive_nn_chunk(df_today, df_before_TX_index ,embeddings, G, average_embedding, transaction_node_features, dict_node=None):

    #Create a container for the new embeddings
    new_embeddings = dict()
    
    #Keep track of the statistics
    stats = {'both':0, 'cardholder':0, 'merchant':0, 'none':0}

    for transaction, transaction_row in tqdm(df_today.iterrows(), total=df_today.shape[0]):
        mms = MinMaxScaler()
        cardholder = transaction_row.CARD_PAN_ID
        merchant = transaction_row.TERM_MIDUID
        
        nearest_neighbor_cardholder = None
        nearest_neighbor_merchant = None

        if G.has_node(cardholder):
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
                embedding_nn_cardholder = embeddings[str(dict_node[str(nearest_neighbor_cardholder)])]
            else:
                embedding_nn_cardholder = embeddings[str(nearest_neighbor_cardholder)]

        if G.has_node(merchant):
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
                embedding_nn_cardholder = embeddings[str(dict_node[str(nearest_neighbor_merchant)])]
            else:
                embedding_nn_merchant = embeddings[str(nearest_neighbor_merchant)]

        if (nearest_neighbor_cardholder != None) & (nearest_neighbor_merchant != None):
            new_embeddings[transaction] = (embedding_nn_cardholder + embedding_nn_merchant)/2
            stats['both'] += 1

        elif nearest_neighbor_cardholder: 
            new_embeddings[transaction] = embedding_nn_cardholder
            stats['cardholder'] += 1

        elif nearest_neighbor_merchant:
            new_embeddings[transaction] = embedding_nn_merchant
            stats['merchant'] += 1

        else:
            new_embeddings[transaction] = average_embedding
            stats['none'] += 1
            
    return new_embeddings, stats
