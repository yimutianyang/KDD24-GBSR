import tensorflow as tf
import numpy as np
import os, pdb, sys
from time import time
from tqdm import tqdm
from shutil import copyfile
import argparse
from rec_dataset import Dataset
from log import Logger
from evaluate import *
from models.GBSR import GBSR_SLightGCN
np.random.seed(2023)
tf.set_random_seed(2023)


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Parameters')
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
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='?')
    parser.add_argument('--beta', type=float, default=5.0, help='cofficient of HSIC regularization')
    parser.add_argument('--sigma', type=float, default=0.25, help='?')
    parser.add_argument('--edge_bias', type=float, default=0.5, help='observation bias of social relations')
    parser.add_argument('--social_noise_ratio', type=float, default=0, help='?')
    return parser.parse_args()


def makir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_file(save_path):
    copyfile('./models/GBSR.py', save_path + 'GBSR.py')
    copyfile('./run_GBSR.py', save_path + 'run_GBSR.py')
    copyfile('./rec_dataset.py', save_path + 'rec_dataset.py')
    copyfile('./evaluate.py', save_path + 'evaluate.py')
    copyfile('./models/PairWise_model.py', save_path + 'PariWise_model.py')


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'yelp':
        args.num_user = 19539
        args.num_item = 22228
        args.lr = 0.001
        args.batch_size = 2048
    elif args.dataset == 'epinions':
        args.num_user = 18202
        args.num_item = 47449
        args.lr = 0.001
        args.batch_size = 2048

    args.data_path = './datasets/' + args.dataset + '/'
    record_path = './saved/' + args.dataset + '/GBSR/' + args.runid + '/'
    model_save_path = record_path + 'models/'
    makir_dir(model_save_path)
    save_file(record_path)
    log = Logger(record_path)
    for arg in vars(args):
        log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    rec_data = Dataset(args)
    rec_model = GBSR_SLightGCN(args, rec_data)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)


    # *****************************  start training  *******************************#
    writer = tf.summary.FileWriter(record_path + '/log/', sess.graph)
    max_hr, max_recall, max_ndcg, early_stop = 0, 0, 0, 0
    topk = args.topk
    ndcg_list = []
    for epoch in range(args.epochs):
        t1 = time()
        data_iter = rec_data._batch_sampling(num_negative=args.num_neg)
        # data_iter = rec_data._uniform_sampling()
        sum_auc, sum_loss1, sum_loss2, sum_loss3, batch_num = 0, 0, 0, 0, 0
        for batch_u, batch_i, batch_j in tqdm(data_iter):
            feed_dict = {rec_model.users: batch_u, rec_model.pos_items: batch_i, rec_model.neg_items: batch_j}
            _auc, _loss1, _loss2, _loss3, _ = sess.run([rec_model.auc, rec_model.ranking_loss,
                                                        rec_model.regu_loss, rec_model.IB_loss, rec_model.opt], feed_dict=feed_dict)
            sum_auc += _auc
            sum_loss1 += _loss1
            sum_loss2 += _loss2
            sum_loss3 += _loss3
            batch_num += 1
        mean_auc = sum_auc / batch_num
        mean_loss1 = sum_loss1 / batch_num
        mean_loss2 = sum_loss2 / batch_num
        mean_loss3 = sum_loss3 / batch_num
        mean_loss = mean_loss1 + mean_loss2
        mean_weight = sess.run(rec_model.masked_gate_input)
        log.write('Mean_weight:{:.4f}\n'.format(mean_weight))
        log.write('Epoch:{:d}, Train_AUC:{:.4f}, Loss_rank:{:.4f}, Loss_reg:{:.4f}, Loss_GIB:{:.4f}\n'
            .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3))
        t2 = time()
        summary_train_loss = sess.run(rec_model.merged_train_loss, feed_dict={rec_model.train_loss: mean_loss,
                                                                              rec_model.train_mf_loss: mean_loss1})
        writer.add_summary(summary_train_loss, epoch)

        # ***************************  Evaluation on Top-20  *****************************#
        if (epoch % 1) == 0:
            early_stop += 1
            user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb])
            hr, recall, ndcg = num_faiss_evaluate(rec_data.valdata, rec_data.traindata,
                                                  [topk], user_matrix, item_matrix,
                                                  rec_data.valdata.keys())  ### all users evaluation
            log.write('Epoch:{:d}, topk:{:d}, R@20:{:.4f}, P@20:{:.4f}, N@20:{:.4f}\n'.format(epoch, topk, recall[topk],
                                                                                              hr[topk], ndcg[topk]))
            rs = sess.run(rec_model.merged_evaluate,
                          feed_dict={rec_model.train_loss: mean_loss, rec_model.train_mf_loss: mean_loss1,
                                     rec_model.recall: recall[topk], rec_model.ndcg: ndcg[topk]})
            writer.add_summary(rs, epoch)
            ndcg_list.append(ndcg[topk])
            max_hr = max(max_hr, hr[topk])
            max_recall = max(max_recall, recall[topk])
            max_ndcg = max(max_ndcg, ndcg[topk])
            if ndcg[topk] == max_ndcg:
            # if recall[topk] == max_recall:
                early_stop = 0
                best_ckpt = 'epoch_' + str(epoch) + '_ndcg_' + str(ndcg[topk]) + '.ckpt'
                saver.save(sess, model_save_path + best_ckpt)
            t3 = time()
            log.write('traintime:{:.4f}, valtime:{:.4f}\n\n'.format(t2 - t1, t3 - t2))
            np.save(record_path + 'ndcg_list.npy', ndcg_list)
            if epoch > 50 and early_stop > args.early_stops:
                log.write('early stop\n')
                log.write('max_recall@20=:{:.4f}, max_ndcg@20=:{:.4f}\n'.format(max_recall, max_ndcg))
                np.save(record_path+'ndcg_list.npy', ndcg_list)
                break

    # ***********************************  start evaluate testdata   ********************************#
    writer.close()
    saver.restore(sess, model_save_path + best_ckpt)
    masked_social_values = sess.run(rec_model.masked_values)
    np.save(record_path + 'social_dropout.npy', masked_social_values)
    masked_social_indices = rec_model.social_index
    np.save(record_path + 'social_edge.npy', masked_social_indices)

    log.write('=================Evaluation results==================\n')
    user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb])
    hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [10, 20, 30, 40, 50], user_matrix, item_matrix,
                                          rec_data.testdata.keys())  ### all users evaluation
    for key in ndcg.keys():
        log.write('Topk:{:3d}, R@20:{:.4f}, P@20:{:.4f} N@20:{:.4f}\n'.format(key, recall[key], hr[key], ndcg[key]))

    log.write('================Sparsity Analysis===================\n')
    u1, u2, u3 = rec_data.user_3group_sparsity()
    user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb])
    _, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [10, 20, 30, 40, 50], user_matrix, item_matrix,
                                          rec_data.testdata.keys())  ### all users evaluation
    print('All evaluation, Recall@20:{:.4f}, NDCG@20:{:.4f}'.format(recall[20], ndcg[20]))
    log.write('Number_G1:{:d}, Number_G2:{:d}, Number_G3:{:d}\n'.format(len(u1), len(u2), len(u3)))
    _, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [10, 20, 30, 40, 50], user_matrix, item_matrix, u1)  ### group1 evaluation
    log.write('Group_U1, Recall@20:{:.4f}, NDCG@20:{:.4f}\n'.format(recall[20], ndcg[20]))
    _, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [10, 20, 30, 40, 50], user_matrix, item_matrix, u2)  ### group2 evaluation
    log.write('Group_U2, Recall@20:{:.4f}, NDCG@20:{:.4f}\n'.format(recall[20], ndcg[20]))
    _, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [10, 20, 30, 40, 50], user_matrix, item_matrix, u3)  ### group3 evaluation
    log.write('Group_U3, Recall@20:{:.4f}, NDCG@20:{:.4f}\n'.format(recall[20], ndcg[20]))
    log.close()
