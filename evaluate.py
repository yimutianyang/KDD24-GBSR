import faiss
import numpy as np
import math
from collections import defaultdict
import pdb
import numba as nb
from numba import prange
from sklearn.metrics import roc_auc_score
import multiprocessing as mp


# nb.config.NUMBA_DEFAULT_NUM_THREADS = 4
@nb.njit()
def compute_ranking_metrics(testusers, testdata, traindata, topk_list, user_rank_pred_items):
    all_metrics = []
    for i in prange(len(testusers)):
        u = testusers[i]
        one_metrics = []
        mask_items = traindata[i]
        test_items = testdata[i]
        pos_length = len(test_items)
        # pred_items找出ranking结果中排除训练样本的前topk个items
        pred_items_all = user_rank_pred_items[u]
        max_length_candicate = len(mask_items) + topk_list[-1]
        pred_items = [item for item in pred_items_all[:max_length_candicate] if item not in mask_items][:topk_list[-1]]
        for topk in topk_list:
            hit_value = 0
            dcg_value = 0
            for idx in prange(topk):
                if pred_items[idx] in test_items:
                    hit_value += 1
                    dcg_value += math.log(2) / math.log(idx + 2)
            target_length = min(topk, pos_length)
            idcg = 0.0
            for k in prange(target_length):
                idcg = idcg + math.log(2) / math.log(k + 2)
            hr_cur = hit_value / target_length
            recall_cur = hit_value / pos_length
            ndcg_cur = dcg_value / idcg
            one_metrics.append([hr_cur, recall_cur, ndcg_cur])
        all_metrics.append(one_metrics)
    return all_metrics

# '''
def num_faiss_evaluate(_test_ratings, _train_ratings, _topk_list, _user_matrix, _item_matrix, _test_users):
    '''
    Evaluation for ranking results
    Topk-largest based on faiss search
    Speeding computation based on numba
    '''
    hr_topk_list = defaultdict(list)
    recall_topk_list = defaultdict(list)
    ndcg_topk_list = defaultdict(list)
    hr_out, recall_out, ndcg_out = {}, {}, {}

    ###  faiss search  ###
    test_users = _test_users #list(_test_ratings.keys())
    query_vectors = _user_matrix
    dim = _user_matrix.shape[-1]
    index = faiss.IndexFlatIP(dim)
    index.add(_item_matrix)
    max_mask_items_length = max(len(_train_ratings[user]) for user in _train_ratings.keys())
    sim, _user_rank_pred_items = index.search(query_vectors, _topk_list[-1] + max_mask_items_length)

    testdata = [list(_test_ratings[user]) for user in test_users]
    # traindata = [list(_train_ratings[user]) if len(_train_ratings[user]) > 0 else [-1] for user in test_users]
    traindata = [list(_train_ratings[user]) if user in _train_ratings.keys() else [-1] for user in test_users]
    all_metrics = compute_ranking_metrics(nb.typed.List(test_users), nb.typed.List(testdata),
                                          nb.typed.List(traindata), nb.typed.List(_topk_list),
                                          nb.typed.List(_user_rank_pred_items))

    ###  output evaluation metrics  ###
    for i, one_metrics in enumerate(all_metrics):
        j = 0
        for topk in _topk_list:
            hr_topk_list[topk].append(one_metrics[j][0])
            recall_topk_list[topk].append(one_metrics[j][1])
            ndcg_topk_list[topk].append(one_metrics[j][2])
            j += 1
    for topk in _topk_list:
        recall_out[topk] = np.mean(recall_topk_list[topk])
        hr_out[topk] = np.mean(hr_topk_list[topk])
        ndcg_out[topk] = np.mean(ndcg_topk_list[topk])
    return hr_out, recall_out, ndcg_out


############################################# Head&Tail evaluation ##############################################
@nb.njit()
def compute_head_tail_ranking_metrics(testusers, testdata, traindata, topk_list, user_rank_pred_items, head_items, tail_items):
    all_metrics = []
    for i in prange(len(testusers)):
        u = testusers[i]
        one_metrics = []
        mask_items = traindata[i]
        test_items = testdata[i]
        pos_length = len(test_items)
        # pred_items找出ranking结果中排除训练样本的前topk个items
        pred_items_all = user_rank_pred_items[u]
        max_length_candicate = len(mask_items) + topk_list[-1]
        pred_items = [item for item in pred_items_all[:max_length_candicate] if item not in mask_items][:topk_list[-1]]
        for topk in topk_list:
            head_hit_value, tail_hit_value = 0, 0
            head_dcg_value, tail_dcg_value = 0, 0
            for idx in prange(topk):
                if pred_items[idx] in test_items:
                    if pred_items[idx] in head_items:
                        head_hit_value += 1
                        head_dcg_value += math.log(2) / math.log(idx + 2)
                    elif pred_items[idx] in tail_items:
                        tail_hit_value += 1
                        tail_dcg_value += math.log(2) / math.log(idx + 2)
                    else:
                        print('without this item')
            target_length = min(topk, pos_length)
            idcg = 0.0
            for k in prange(target_length):
                idcg = idcg + math.log(2) / math.log(k + 2)
            head_hr_cur = head_hit_value / target_length
            head_recall_cur = head_hit_value / pos_length
            head_ndcg_cur = head_dcg_value / idcg
            tail_hr_cur = tail_hit_value / target_length
            tail_recall_cur = tail_hit_value / pos_length
            tail_ndcg_cur = tail_dcg_value / idcg
            one_metrics.append([head_hr_cur, head_recall_cur, head_ndcg_cur,
                                tail_hr_cur, tail_recall_cur, tail_ndcg_cur])
        all_metrics.append(one_metrics)
    return all_metrics

# '''
def num_faiss_evaluate_head_tail(_test_ratings, _train_ratings, _topk_list, _user_matrix, _item_matrix, _test_users, _head_items, _tail_items):
    '''
    Evaluation for ranking results
    Topk-largest based on faiss search
    Speeding computation based on numba
    '''
    hr_topk_list_h = defaultdict(list)
    recall_topk_list_h = defaultdict(list)
    ndcg_topk_list_h = defaultdict(list)
    hr_topk_list_t = defaultdict(list)
    recall_topk_list_t = defaultdict(list)
    ndcg_topk_list_t = defaultdict(list)
    hr_out_h, recall_out_h, ndcg_out_h = {}, {}, {}
    hr_out_t, recall_out_t, ndcg_out_t = {}, {}, {}

    ###  faiss search  ###
    test_users = _test_users #list(_test_ratings.keys())
    query_vectors = _user_matrix
    dim = _user_matrix.shape[-1]
    index = faiss.IndexFlatIP(dim)
    index.add(_item_matrix)
    max_mask_items_length = max(len(_train_ratings[user]) for user in _train_ratings.keys())
    sim, _user_rank_pred_items = index.search(query_vectors, _topk_list[-1] + max_mask_items_length)

    testdata = [list(_test_ratings[user]) for user in test_users]
    traindata = [list(_train_ratings[user]) if len(_train_ratings[user]) > 0 else [-1] for user in test_users]
    # traindata = [list(_train_ratings[user]) if user in _train_ratings.keys() else [-1] for user in test_users]
    all_metrics = compute_head_tail_ranking_metrics(nb.typed.List(test_users), nb.typed.List(testdata),
                                          nb.typed.List(traindata), nb.typed.List(_topk_list),
                                          nb.typed.List(_user_rank_pred_items), nb.typed.List(_head_items),
                                          nb.typed.List(_tail_items))

    ###  output evaluation metrics  ###
    for i, one_metrics in enumerate(all_metrics):
        j = 0
        for topk in _topk_list:
            hr_topk_list_h[topk].append(one_metrics[j][0])
            recall_topk_list_h[topk].append(one_metrics[j][1])
            ndcg_topk_list_h[topk].append(one_metrics[j][2])
            hr_topk_list_t[topk].append(one_metrics[j][3])
            recall_topk_list_t[topk].append(one_metrics[j][4])
            ndcg_topk_list_t[topk].append(one_metrics[j][5])
            j += 1
    for topk in _topk_list:
        recall_out_h[topk] = np.mean(recall_topk_list_h[topk])
        hr_out_h[topk] = np.mean(hr_topk_list_h[topk])
        ndcg_out_h[topk] = np.mean(ndcg_topk_list_h[topk])
        recall_out_t[topk] = np.mean(recall_topk_list_t[topk])
        hr_out_t[topk] = np.mean(hr_topk_list_t[topk])
        ndcg_out_t[topk] = np.mean(ndcg_topk_list_t[topk])
    return recall_out_h, ndcg_out_h, recall_out_t, ndcg_out_t