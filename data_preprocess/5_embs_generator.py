import sys
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np

def main():
    print("Generating category embeddings with BERT...")
    data_name = sys.argv[1]

    data_path1 = "./tmp/{}_cid2_dict.txt".format(data_name)
    data_path2 = "./tmp/{}_cid3_dict.txt".format(data_name)
    # load BERT pretrained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embedding_matrix_list = []
    index = 2
    for data_path in [data_path1, data_path2]:
        id_pd = pd.read_csv(data_path, sep='\t', header=None)
        id_pd = id_pd.sort_values(by=0)
        categories = np.array(id_pd[1])

        embeddings_matrix = []
        for text in categories:
            # tokenizer
            tokens = tokenizer.encode(text, add_special_tokens=True)
            tokens_tensor = torch.tensor([tokens])

            with torch.no_grad():
                outputs = model(tokens_tensor)

            # obtain embeddings
            last_hidden_states = outputs[0]
            embeddings = torch.mean(last_hidden_states, dim=1)
            embeddings_matrix.append(embeddings)
        embeddings_matrix = torch.cat(embeddings_matrix, dim=0)
        embedding_matrix_list.append(embeddings_matrix)
        print("category {} embedding shape: {}".format(index, embeddings_matrix.shape))
        index += 1
    save_path = "./embs/{}_embeddings.npz".format(data_name)
    np.savez(save_path, cid2_emb=embedding_matrix_list[0], cid3_emb=embedding_matrix_list[1])

if __name__ == '__main__':
    main()

