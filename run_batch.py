import numpy as np

#from model_final import READ_wo_mid_bert, READ_wo_att_bert, READ_bert
import torch

from model_final2 import READ
from utils import *
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,average_precision_score
import random
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='READ-GNN')
parser.add_argument('--ckpt_path', type=str, default='None')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='Appliances')
parser.add_argument('--num_layer', type=int, default='1')
parser.add_argument('--train_ratio', type=str, default='40')
parser.add_argument('--neg_num', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--weight_decay', type=float, default=5e-8)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--embedding_dim', type=int, default=16)
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--val_epoch', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32768000)
parser.add_argument('--mode', type=int, default=3)

args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device)
    num_layer = args.num_layer
    neg_num = args.neg_num
    batch_size = args.batch_size
    mode = args.mode
    if args.ckpt_path == 'None':
        ckpt_path = 'checkpoints/' + str(args.dataset)
    else:
        ckpt_path = 'checkpoints/' + str(args.ckpt_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)


    val_topk = 150
    test_topk = 350

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = [i + 1 for i in range(args.runs)]

    '''load dataset'''
    features, price_bin, com_edge_index, sim_edge_index, train_set, val_set, test_set = load_dataset_emb(args.dataset)

    num_items = features.shape[0]
    #feature_size = features.shape[1]
    feature_size = 5000

    adj, adj_ori = generate_adj(com_edge_index, num_items)

    features = torch.FloatTensor(features).to(device)
    price_bin = torch.LongTensor(price_bin).to(device)
    adj = adj.to(device)

    train_set = torch.LongTensor(train_set).to(device)
    val_set = torch.LongTensor(val_set).to(device)
    test_set = torch.LongTensor(test_set).to(device)

    mean_mrr = []
    mean_hr5 = []
    mean_hr10 = []
    mean_ndcg = []
    mean_cov10 = []
    mean_val_auc = []
    mean_test_auc = []
    mean_val_prauc = []
    mean_test_prauc = []
    mean_val_f1 = []
    mean_test_f1 = []
    mean_val_recall = []
    mean_test_recall = []
    mean_val_precision = []
    mean_test_precision = []
    all_test_score = []
    all_test_label = []
    scaler = torch.cuda.amp.GradScaler()

    ''''train the model'''
    for run in range(args.runs):
        seed = seeds[run]
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        '''
        if mode == 3:
            model = READ_bert(feature_size, args.embedding_dim)
        elif mode == 2:
            model = READ_wo_att_bert(feature_size, args.embedding_dim)
        else:
            model = READ_wo_mid_bert(feature_size, args.embedding_dim)
        '''
        model = READ(feature_size, args.embedding_dim, 20, mode)

        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        cnt_wait = 0
        best_epoch = 0
        best_mrr = 0
        best_hr5 = 0
        best_hr10 = 0
        best_ndcg = 0
        best_cov10 = 0
        for epoch in range(args.num_epoch):

            model.train()
            total_loss = 0.

            train_dataset = Train_Dataset(train_set)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            num_batch = len(train_dataloader)

            for batch_idx, sample in enumerate(tqdm(train_dataloader)):
                optimiser.zero_grad()
                train_batch_set = sample
                loss = model(features, price_bin, adj, train_batch_set)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()

            mean_loss = total_loss/num_batch

            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)


            #validation
            if 1:
                if epoch % args.val_epoch == 0:
                    with torch.no_grad():
                        model.eval()
                        mrr_score, hr5_score, hr10_score, ndcg_score, cov10_score = model.inference(features, price_bin, adj, val_set)
                        mrr_score = mrr_score.to(torch.device('cpu'))
                        hr5_score = hr5_score.to(torch.device('cpu'))
                        hr10_score = hr10_score.to(torch.device('cpu'))
                        #ndcg_score = ndcg_score.to(torch.device('cpu'))
                        #cov10_score = cov10_score.to(torch.device('cpu'))

                    print('Epoch:{} Val MRR:{:.4f}, HR@5:{:.4f}, HR@10:{:.4f} NDCG:{:.4f} COV10:{:.4f}'.format(
                        epoch, mrr_score, hr5_score,hr10_score, ndcg_score, cov10_score), flush=True)

                    if ndcg_score > best_ndcg:
                        best_mrr = mrr_score
                        best_hr5 = hr5_score
                        best_hr10 = hr10_score
                        best_ndcg = ndcg_score
                        best_cov10 = cov10_score
                        best_run = run
                        best_epoch = epoch
                        cnt_wait = 0
                        torch.save(model.state_dict(), '{}/model.pkl'.format(ckpt_path))
                    else:
                        cnt_wait += 1

                    if cnt_wait == args.patience:
                        print('Early stopping!', flush=True)
                        break

        # Testing
        if 1:
            print('Loading {}th epoch'.format(best_epoch), flush=True)
            model.load_state_dict(torch.load('{}/model.pkl'.format(ckpt_path)))
            print('Testing AUC!', flush=True)
            with torch.no_grad():
                model.eval()
                mrr_score, hr5_score, hr10_score, ndcg_score, cov10_score = model.inference(features, price_bin, adj, test_set)
                mrr_score = mrr_score.to(torch.device('cpu'))
                hr5_score = hr5_score.to(torch.device('cpu'))
                hr10_score = hr10_score.to(torch.device('cpu'))
                #recommend_idx = recommend_idx.to(torch.device('cpu'))
                #ndcg_score = ndcg_score.to(torch.device('cpu'))
                #cov10_score = cov10_score.to(torch.device('cpu'))
                '''
                adj_low = np.array(adj_low.to(torch.device('cpu')))
                adj_mid = np.array(adj_mid.to(torch.device('cpu')))
                adj_ori = np.array(adj.to(torch.device('cpu')).to_dense())
                adj_low = pd.DataFrame(np.array(adj_low))
                adj_mid = pd.DataFrame(np.array(adj_mid))
                adj_ori = pd.DataFrame(np.array(adj_ori))

                adj_low.to_csv('adj_low.csv', index=False)
                adj_mid.to_csv('adj_mid.csv', index=False)
                adj_ori.to_csv('adj_ori.csv', index=False)
                '''
                #scores = np.array(scores.to(torch.device('cpu')))
                #scores = pd.DataFrame(np.array(scores))
                #scores.to_csv('test_scores.csv', index=False)
                #recommend_idx = pd.DataFrame(np.array(recommend_idx))
                #recommend_idx.to_csv('recommend_idx' + str(args.dataset)+ '.csv', index=False)


            mean_mrr.append(mrr_score)
            mean_hr5.append(hr5_score)
            mean_hr10.append(hr10_score)
            mean_ndcg.append(ndcg_score)
            mean_cov10.append(cov10_score)
            print('Testing MRR:{:.4f}, HR@5:{:.4f}, HR@10:{:.4f}, NDCG:{:.4f}, COV10:{:.4f}'.format(mrr_score, hr5_score, hr10_score, ndcg_score
                                                                                                    , cov10_score),flush=True)


    print("--------------------final results--------------------")
    print('Test MRR: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_mrr)/len(mean_mrr), np.std(mean_mrr),
        max(mean_mrr), min(mean_mrr)), flush=True)
    print('Test HR@5: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_hr5)/len(mean_hr5), np.std(mean_hr5),
        max(mean_hr5), min(mean_hr5)), flush=True)
    print('Test HR@10: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_hr10)/len(mean_hr10), np.std(mean_hr10),
        max(mean_hr10), min(mean_hr10)), flush=True)
    print('Test NDCG: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_ndcg)/len(mean_ndcg), np.std(mean_ndcg),
        max(mean_ndcg), min(mean_ndcg)), flush=True)
    print('Test COV10: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_cov10)/len(mean_cov10), np.std(mean_cov10),
        max(mean_cov10), min(mean_cov10)), flush=True)