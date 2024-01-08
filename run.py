from model import SComGNN
from utils import *
import random
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='SComGNN')
parser.add_argument('--ckpt_path', type=str, default='None')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='Appliances')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=5e-8)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--embedding_dim', type=int, default=16)
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--val_epoch', type=int, default=1)
parser.add_argument('--mode', choices=["att", "concat", "mid", "low"], help='the version of models', default='att')

args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device)

    ckpt_path = "checkpoints/"
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    if args.ckpt_path == 'None':
        ckpt_path = "checkpoints/{}".format(args.dataset)
    else:
        ckpt_path = "checkpoints/{}".format(args.ckpt_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = [i + 1 for i in range(args.runs)]

    #load dataset
    features, price_bin, com_edge_index, train_set, val_set, test_set = load_dataset(args.dataset)

    num_items = features.shape[0]
    adj = generate_adj(com_edge_index, num_items)

    features = torch.FloatTensor(features).to(device)
    price_bin = torch.LongTensor(price_bin).to(device)
    adj = adj.to(device)
    train_set = torch.LongTensor(train_set).to(device)
    val_set = torch.LongTensor(val_set).to(device)
    test_set = torch.LongTensor(test_set).to(device)

    mean_hr5 = []
    mean_hr10 = []
    mean_ndcg = []

    #train the model
    for run in range(args.runs):
        seed = seeds[run]
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        model = SComGNN(args.embedding_dim, 20, args.mode)

        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        cnt_wait = 0
        best_epoch = 0
        best_hr5 = 0
        best_hr10 = 0
        best_ndcg = 0

        for epoch in range(args.num_epoch):

            model.train()
            optimiser.zero_grad()
            loss = model(features, price_bin, adj, train_set)
            loss.backward()
            optimiser.step()
            print('Epoch:{} Loss:{:.8f}'.format(epoch, loss.item()), flush=True)

            #validation
            if 1:
                if epoch % args.val_epoch == 0:
                    with torch.no_grad():
                        model.eval()
                        hr5_score, hr10_score, ndcg_score= model.inference(features, price_bin, adj, val_set)

                        hr5_score = hr5_score.to(torch.device('cpu'))
                        hr10_score = hr10_score.to(torch.device('cpu'))

                    print('Epoch:{} Val HR@5:{:.4f}, HR@10:{:.4f} NDCG:{:.4f}'.format(epoch, hr5_score, hr10_score, ndcg_score), flush=True)

                    if ndcg_score > best_ndcg:
                        best_hr5 = hr5_score
                        best_hr10 = hr10_score
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
                hr5_score, hr10_score, ndcg_score= model.inference(features, price_bin, adj, test_set)
                hr5_score = hr5_score.to(torch.device('cpu'))
                hr10_score = hr10_score.to(torch.device('cpu'))

            mean_hr5.append(hr5_score)
            mean_hr10.append(hr10_score)
            mean_ndcg.append(ndcg_score)
            print('Testing HR@5:{:.4f}, HR@10:{:.4f}, NDCG:{:.4f}'.format(hr5_score, hr10_score, ndcg_score),flush=True)


    print("--------------------final results--------------------")
    print('Test HR@5: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_hr5)/len(mean_hr5), np.std(mean_hr5),
        max(mean_hr5), min(mean_hr5)), flush=True)
    print('Test HR@10: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_hr10)/len(mean_hr10), np.std(mean_hr10),
        max(mean_hr10), min(mean_hr10)), flush=True)
    print('Test NDCG: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_ndcg)/len(mean_ndcg), np.std(mean_ndcg),
        max(mean_ndcg), min(mean_ndcg)), flush=True)