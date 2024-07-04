import os, pdb, sys
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import argparse
from shutil import copyfile
from time import time
from tqdm import tqdm
sys.path.append('./')
sys.path.append('../')
from set import *
from evaluate import *
from log import Logger
from GBSR import GBSR
from rec_dataset import Dataset



def parse_args():
    parser = argparse.ArgumentParser(description='GBSR Parameters')
    ### general parameters ###
    parser.add_argument('--dataset', type=str, default='douban_book', help='?')
    parser.add_argument('--runid', type=str, default='0', help='current log id')
    parser.add_argument('--device_id', type=str, default='0', help='?')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--topk', type=int, default=20, help='Topk value for evaluation')   # NDCG@20 as convergency metric
    parser.add_argument('--early_stops', type=int, default=10, help='model convergent when NDCG@20 not increase for x epochs')
    parser.add_argument('--num_neg', type=int, default=1, help='number of negetiva samples for each [u,i] pair')

    ### model parameters ###
    parser.add_argument('--gcn_layer', type=int, default=3, help='?')
    parser.add_argument('--num_user', type=int, default=13024, help='max uid')
    parser.add_argument('--num_item', type=int, default=22347, help='max iid')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent embedding dimension')
    parser.add_argument('--init_type', type=str, default='norm', help='?')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='?')
    parser.add_argument('--beta', type=float, default=5.0, help='?')
    parser.add_argument('--sigma', type=float, default=0.25, help='?')
    parser.add_argument('--edge_bias', type=float, default=0.5, help='observation bias of social relations')
    parser.add_argument('--social_noise_ratio', type=float, default=0, help='?')
    return parser.parse_args()


def makir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_file(save_path):
    copyfile('./GBSR.py', save_path + 'GBSR.py')
    copyfile('./run_GBSR.py', save_path + 'run_GBSR.py')
    copyfile('../rec_dataset.py', save_path + 'rec_dataset.py')

def eval_test(model):
    model.eval()
    with torch.no_grad():
        masked_adj_matrix = model.graph_learner(model.user_embeddings)
        user_emb, item_emb = model.forward(masked_adj_matrix)
    return user_emb.cpu().detach().numpy(), item_emb.cpu().detach().numpy()


if __name__ == '__main__':
    seed_everything(2023)
    args = parse_args()
    if args.dataset == 'yelp':
        args.num_user = 19539
        args.num_item = 22228
    elif args.dataset == 'epinions':
        args.num_user = 18202
        args.num_item = 47449

    args.data_path = '../datasets/' + args.dataset + '/'
    record_path = '../saved/' + args.dataset + '/GBSR/' + args.runid + '/'
    model_save_path = record_path + 'models/'
    makir_dir(model_save_path)
    save_file(record_path)
    log = Logger(record_path)
    for arg in vars(args):
        log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    rec_data = Dataset(args)
    rec_model = GBSR(args, rec_data)
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
    rec_model.to(device)
    optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr)

    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    max_hr, max_recall, max_ndcg, early_stop = 0, 0, 0, 0
    topk = args.topk
    best_epoch = 0

    model_files = []
    max_to_keep = 5

    for epoch in tqdm(range(args.epochs), desc=set_color(f"Train:", 'pink'), colour='yellow',
                      dynamic_ncols=True, position=0):
        t1 = time()
        sum_auc, all_rank_loss, all_reg_loss, all_ib_loss, all_total_loss, batch_num = 0, 0, 0, 0, 0, 0
        rec_model.train()
        #  batch数据
        loader = rec_data._batch_sampling(num_negative=args.num_neg)
        for u, i, j in tqdm(loader, desc='All_batch'):
            u = torch.tensor(u).type(torch.long).to(device)  # [batch_size]
            i = torch.tensor(i).type(torch.long).to(device)  # [batch_size]
            j = torch.tensor(j).type(torch.long).to(device)  # [batch_size]
            auc, rank_loss, reg_loss, ib_loss, total_loss = rec_model.calculate_all_loss(u, i, j)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            sum_auc += auc.item()
            all_rank_loss += rank_loss.item()
            all_reg_loss += reg_loss.item()
            all_ib_loss += ib_loss.item()
            all_total_loss += total_loss.item()
            batch_num += 1
        mean_auc = sum_auc / batch_num
        mean_rank_loss = all_rank_loss / batch_num
        mean_reg_loss = all_reg_loss / batch_num
        mean_ib_loss = all_ib_loss / batch_num
        mean_total_loss = all_total_loss / batch_num
        log.write(set_color('Epoch:{:d}, Train_AUC:{:.4f}, Loss_rank:{:.4f}, Loss_reg:{:.4f}, Loss_ib:{:.4f}, Loss_sum:{:.4f}\n'
                            .format(epoch, mean_auc, mean_rank_loss, mean_reg_loss, mean_ib_loss, mean_total_loss), 'blue'))
        t2 = time()


        # ***************************  evaluation on Top-20  *****************************#
        if epoch % 1 == 0:
            early_stop += 1
            user_emb, item_emb = eval_test(rec_model)
            hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata, [20], user_emb, item_emb,
                                        rec_data.testdata.keys())
            if ndcg[20] >= max_ndcg or ndcg[20] == max_ndcg and recall[20] >= max_recall:
                best_epoch = epoch
                max_hr = hr[20]
                max_recall = recall[20]
                max_ndcg = ndcg[20]
            log.write(set_color(
                'Current Evaluation: Epoch:{:d},  topk:{:d}, recall:{:.4f}, ndcg:{:.4f}\n'.format(epoch, topk,
                                                                                                  recall[20], ndcg[20]),
                'green'))
            log.write(set_color(
                'Best Evaluation: Epoch:{:d},  topk:{:d}, recall:{:.4f}, ndcg:{:.4f}\n'.format(best_epoch, topk,
                                                                                               max_recall, max_ndcg),
                'red'))

            if ndcg[20] == max_ndcg:
                early_stop = 0
                best_ckpt = 'epoch_' + str(epoch) + '_ndcg_' + str(ndcg[20]) + '.ckpt'
                filepath = model_save_path + best_ckpt
                torch.save(rec_model.state_dict(), filepath)
                print(f"Saved model to {filepath}")
                model_files.append(filepath)
                if len(model_files) > max_to_keep:
                    oldest_file = model_files.pop(0)
                    os.remove(oldest_file)
                    print(f"Removed old model file: {oldest_file}")

            t3 = time()
            log.write('traintime:{:.4f}, valtime:{:.4f}\n\n'.format(t2 - t1, t3 - t2))
            if epoch > 50 and early_stop > args.early_stops:
                log.write('early stop: ' + str(epoch) + '\n')
                log.write(set_color('max_recall@20=:{:.4f}, max_ndcg@20=:{:.4f}\n'.format(max_recall, max_ndcg), 'green'))
                break

    # ***********************************  start evaluate testdata   ********************************#
    rec_model.load_state_dict(torch.load(model_save_path + best_ckpt))
    user_emb, item_emb = eval_test(rec_model)
    hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], user_emb, item_emb,
                                          rec_data.testdata.keys())
    for key in ndcg.keys():
        log.write(set_color(
            'Topk:{:3d}, HR:{:.4f}, Recall:{:.4f}, NDCG:{:.4f}\n'.format(key, hr[key], recall[key], ndcg[key]), 'cyan'))
    log.close()
    print('END')
