from transformers import BertTokenizer, BertModel
from utils import *
import argparse

parser = argparse.ArgumentParser(description='GCO')
parser.add_argument('--dataset', type=str, default='Appliances')
args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset))
    dataset = args.dataset
    data_path1 = './baseline_model/CIKM2020_DecGCN/preprocessing/tmp/' + dataset + '_cid2_dict.txt'
    data_path2 = './baseline_model/CIKM2020_DecGCN/preprocessing/tmp/' + dataset + '_cid3_dict.txt'
    # 加载BERT预训练模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embedding_matrix_list = []
    for data_path in [data_path1, data_path2]:
        id_pd = pd.read_csv(data_path, sep='\t', header=None)
        id_pd = id_pd.sort_values(by=0)
        categories = np.array(id_pd[1])

        # 定义文本
        embeddings_matrix = []
        for text in categories:
            # 对文本进行分词和编码
            tokens = tokenizer.encode(text, add_special_tokens=True)
            tokens_tensor = torch.tensor([tokens])

            # 计算BERT模型的输出
            with torch.no_grad():
                outputs = model(tokens_tensor)

            # 提取嵌入向量
            last_hidden_states = outputs[0]
            embeddings = torch.mean(last_hidden_states, dim=1)
            embeddings_matrix.append(embeddings)
        embeddings_matrix = torch.cat(embeddings_matrix, dim=0)
        embedding_matrix_list.append(embeddings_matrix)
        print("emb shape:", embeddings_matrix.shape)
    save_path = './dataset/processed_val/' + dataset + '_embeddings.npz'
    np.savez(save_path, cid2_emb=embedding_matrix_list[0], cid3_emb=embedding_matrix_list[1])
    print("finished")

