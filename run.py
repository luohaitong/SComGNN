from model import READ
from utils import *
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,average_precision_score
import random
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='READ-GNN')
parser.add_argument('--ckpt_path', type=str, default='None')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='Appliances')
parser.add_argument('--num_layer', type=int, default='1')
parser.add_argument('--train_ratio', type=str, default='40')
parser.add_argument('--neg_num', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-8)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--embedding_dim', type=int, default=16)
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--val_epoch', type=int, default=1)


args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device)
    num_layer = args.num_layer
    neg_num = args.neg_num
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
    features, com_edge_index, train_set, val_set, test_set = load_dataset3(args.dataset)

    num_items = features.shape[0]
    feature_size = features.shape[1]

    adj = generate_adj(com_edge_index, num_items)

    features = torch.FloatTensor(features).to(device)
    adj = adj.to(device)

    train_set = torch.LongTensor(train_set).to(device)
    val_set = torch.LongTensor(val_set).to(device)
    test_set = torch.LongTensor(test_set).to(device)

    mean_mrr = []
    mean_hr = []
    mean_ndcg = []
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

        model = READ(feature_size, args.embedding_dim, num_layer)

        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        cnt_wait = 0
        best_epoch = 0
        best_mrr = 0
        best_hr = 0
        best_ndcg = 0
        for epoch in range(args.num_epoch):

            model.train()
            optimiser.zero_grad()
            loss, mrr_score, hr_score, ndcg_score = model(features, adj, train_set)
            loss.backward()
            optimiser.step()

            #print('Epoch:{} Loss:{:.8f} MRR:{:.4f} HR:{:.4f} NDCG:{:.4f}'.format(epoch, loss.item(), mrr_score.item(),
                                                                       #hr_score.item(), ndcg_score.item()), flush=True)
            print('Epoch:{} Loss:{:.8f}'.format(epoch, loss.item()), flush=True)
            #validation
            if 1:
                if epoch % args.val_epoch == 0:
                    with torch.no_grad():
                        model.eval()
                        mrr_score, hr_score, ndcg_score = model.inference(features, adj, val_set)
                        mrr_score = mrr_score.to(torch.device('cpu'))
                        hr_score = hr_score.to(torch.device('cpu'))
                        ndcg_score = ndcg_score.to(torch.device('cpu'))

                    print('Epoch:{} Val MRR:{:.4f}, HR:{:.4f}, NDCG:{:.4f}'.format(
                        epoch, mrr_score, hr_score, ndcg_score), flush=True)

                    if hr_score > best_hr:
                        best_mrr = mrr_score
                        best_hr = hr_score
                        best_ndcg = ndcg_score
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
                mrr_score, hr_score, ndcg_score = model.inference(features, adj, test_set)
                mrr_score = mrr_score.to(torch.device('cpu'))
                hr_score = hr_score.to(torch.device('cpu'))
                ndcg_score = ndcg_score.to(torch.device('cpu'))

            mean_mrr.append(mrr_score)
            mean_hr.append(hr_score)
            mean_ndcg.append(ndcg_score)
            print('Testing MRR:{:.4f}, HR:{:.4f}, NDCG:{:.4f}'.format(mrr_score, hr_score, ndcg_score),flush=True)


    print("--------------------final results--------------------")
    print('Test MRR: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_mrr)/len(mean_mrr), np.std(mean_mrr),
        max(mean_mrr), min(mean_mrr)), flush=True)
    print('Test HR: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_hr)/len(mean_hr), np.std(mean_hr),
        max(mean_hr), min(mean_hr)), flush=True)
    print('Test NDCG: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_ndcg)/len(mean_ndcg), np.std(mean_ndcg),
        max(mean_ndcg), min(mean_ndcg)), flush=True)



