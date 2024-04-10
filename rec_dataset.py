import numpy as np
import os, pdb
from time import time
import scipy.sparse as sp
import numba as nb
from collections import defaultdict
import random
import argparse
import json
from sklearn.decomposition import PCA


@nb.njit()
def negative_sampling(training_user, training_item, traindata, num_item, num_negative):
    '''
    return: [u,i,j] for training, u interacted with i, not interacted with j
    '''
    trainingData = []
    for k in range(len(training_user)):
        u = training_user[k]
        pos_i = training_item[k]
        for _ in range(num_negative):
            neg_j = random.randint(0, num_item - 1)
            while neg_j in traindata[u]:
                neg_j = random.randint(0, num_item - 1)
            trainingData.append([u, pos_i, neg_j])
    return np.array(trainingData)


@nb.njit()
def Uniform_sampling(batch_users, traindata, num_item):
    trainingData = []
    for u in batch_users:
        pos_items = traindata[u]
        pos_id = np.random.randint(low=0, high=len(pos_items), size=1)[0]
        pos_item = pos_items[pos_id]
        neg_item = random.randint(0, num_item - 1)
        while neg_item in pos_items:
            neg_item = random.randint(0, num_item - 1)
        trainingData.append([u, pos_item, neg_item])
    return np.array(trainingData)


class Dataset(object):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.num_node = self.num_user + self.num_item
        self.batch_size = args.batch_size
        self.social_noise_ratio = args.social_noise_ratio
        ### load and process dataset ###
        self.load_data()
        self.data_to_numba_dict()
        self.training_user, self.training_item = [], []
        for u, items in self.traindata.items():
            self.training_user.extend([u] * len(items))
            self.training_item.extend(items)
        self.adj_matrix = self.lightgcn_adj_matrix()
        try:
            self.uu_i_matrix = self.social_lightgcn_adj_matrix()
        except:
            pass


    def load_data(self):
        self.traindata = np.load(self.data_path + 'traindata.npy', allow_pickle=True).tolist()
        self.valdata = np.load(self.data_path + 'testdata.npy', allow_pickle=True).tolist()
        self.testdata = np.load(self.data_path + 'testdata.npy', allow_pickle=True).tolist()
        try:
            if self.social_noise_ratio == 0:
                self.user_users = np.load(self.data_path + 'user_users_d.npy', allow_pickle=True).tolist()
            elif self.social_noise_ratio == 0.2:
                self.user_users = np.load(self.data_path + 'attacked_user_users_0.2.npy', allow_pickle=True).tolist()
            elif self.social_noise_ratio == 0.5:
                self.user_users = np.load(self.data_path + 'attacked_user_users_0.5.npy', allow_pickle=True).tolist()
            elif self.social_noise_ratio == 1.0:
                self.user_users = np.load(self.data_path + 'attacked_user_users_1.0.npy', allow_pickle=True).tolist()
            elif self.social_noise_ratio == 2.0:
                self.user_users = np.load(self.data_path + 'attacked_user_users_2.0.npy', allow_pickle=True).tolist()
            self.social_i, self.social_j = [], []
            for u, users in self.user_users.items():
                self.social_i.extend([u] * len(users))
                self.social_j.extend(users)
            print('successfull load social networks')
            # attacked_user_users = self.add_noisy_social_links(ratio=0.2)
            # np.save(self.data_path + 'attacked_user_users_0.2.npy', attacked_user_users)
        except:
            pass


    def user_3group_sparsity(self, count_list=[10, 50]):
        u1, u2, u3 = [], [], []
        for u in self.testdata.keys():
            if u in self.traindata.keys():
                items = self.traindata[u]
            else:
                continue
            if len(items) < count_list[0]:
                u1.append(u)
            elif len(items) < count_list[1]:
                u2.append(u)
            else:
                u3.append(u)
        return u1, u2, u3


    def add_noisy_social_links(self, ratio=0.5):
        attacked_social_relations = {}
        for u, users in self.user_users.items():
            attacked_social_relations[u] = list(users)
            for _ in range(int(len(users)*ratio)):
                fake_u = random.randint(0, self.num_user-1)
                while fake_u in attacked_social_relations[u]:
                    fake_u = random.randint(0, self.num_user - 1)
                attacked_social_relations[u].append(fake_u)
        return attacked_social_relations



    def data_to_numba_dict(self):
        self.traindict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.traindata.items():
            if len(values) > 0:
                self.traindict[key] = np.asarray(list(values))

        self.valdict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.valdata.items():
            if len(values) > 0:
                self.valdict[key] = np.asarray(list(values))

        self.testdict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.testdata.items():
            if len(values) > 0:
                self.testdict[key] = np.asarray(list(values))


    def social_lightgcn_adj_matrix(self):
        '''
        return: sparse adjacent matrix, refer lightgcn
        '''
        user_np = np.array(self.training_user)
        item_np = np.array(self.training_item)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        social_i = np.array(self.social_i)
        social_j = np.array(self.social_j)
        social_r = np.ones_like(social_i, dtype=np.float32)
        social_adj = sp.csr_matrix((social_r, (social_i, social_j)), shape=(self.num_node, self.num_node))
        # pdb.set_trace()
        adj_mat = adj_mat + social_adj

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix


    def social_index_in_social_lightgcn(self):
        uu_i_matrix = self.social_lightgcn_adj_matrix()
        adj_indices, adj_values, adj_shape = self.convert_csr_to_sparse_tensor_inputs(uu_i_matrix)
        social_edge_index = []
        for i in range(adj_indices.shape[0]):
            # pdb.set_trace()
            if adj_indices[i, 0] < self.num_user:
                if adj_indices[i, 1] < self.num_user:
                    social_edge_index.append(i)
                else:
                    continue
            else:
                continue
        # pdb.set_trace()
        print('number of social edges:', len(social_edge_index))
        return np.array(social_edge_index)



    def lightgcn_adj_matrix(self):
        '''
        return: sparse adjacent matrix, refer lightgcn
        '''
        user_np = np.array(self.training_user)
        item_np = np.array(self.training_item)
        # pdb.set_trace()
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix


    def convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        # pdb.set_trace()
        return indices, coo.data, coo.shape


    def _uniform_sampling(self):
        batch_num = int(len(self.training_user) / self.batch_size) + 1
        for _ in range(batch_num):
            batch_users = random.sample(list(self.traindata.keys()), self.batch_size)
            batch_data = Uniform_sampling(nb.typed.List(batch_users), self.traindict, self.num_item)
            yield batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]


    def _batch_sampling(self, num_negative):
        t1 = time()
        ### 三元组采样使用numba加速
        triplet_data = negative_sampling(nb.typed.List(self.training_user), nb.typed.List(self.training_item),
                                         self.traindict, self.num_item, num_negative)
        print('prepare training data cost time:{:.4f}'.format(time() - t1))
        batch_num = int(len(triplet_data) / self.batch_size) + 1
        indexs = np.arange(triplet_data.shape[0])
        np.random.shuffle(indexs)
        for k in range(batch_num):
            index_start = k * self.batch_size
            index_end = min((k + 1) * self.batch_size, len(indexs))
            if index_end == len(indexs):
                index_start = len(indexs) - self.batch_size
            batch_data = triplet_data[indexs[index_start:index_end]]
            yield batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]