import torch
import torch.nn as nn
import torch.nn.functional as F


def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)    ### real_kernel

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2


class GBSR(nn.Module):
    def __init__(self, args, dataset):
        super(GBSR, self).__init__()
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.gcn_layer = args.gcn_layer
        self.latent_dim = args.latent_dim
        self.init_type = args.init_type
        self.l2_reg = args.l2_reg
        self.beta = args.beta
        self.sigma = args.sigma
        self.edge_bias = args.edge_bias
        self.batch_size = args.batch_size
        self.adj_matrix, self.social_index = dataset.get_uu_i_matrix()
        self.social_u = self.adj_matrix.indices()[0][self.social_index]
        self.social_v = self.adj_matrix.indices()[1][self.social_index]
        self.social_weight = self.adj_matrix.values()[self.social_index]
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self._init_weights()


    def _init_weights(self):
        self.user_embeddings = nn.Embedding(self.num_user, self.latent_dim)
        self.item_embeddings = nn.Embedding(self.num_item, self.latent_dim)
        if self.init_type == 'norm':
            nn.init.normal_(self.user_embeddings.weight, std=0.01)
            nn.init.normal_(self.item_embeddings.weight, std=0.01)
        elif self.init_type == 'xa_norm':
            nn.init.xavier_normal_(self.user_embeddings.weight)
            nn.init.xavier_normal_(self.item_embeddings.weight)
        else:
            raise NotImplementedError
        self.activate = nn.ReLU()
        self.linear_1 = nn.Linear(in_features=2*self.latent_dim, out_features=self.latent_dim, bias=True)
        self.linear_2 = nn.Linear(in_features=self.latent_dim, out_features=1, bias=True)
        return None


    def graph_learner(self, user_emb):
        row, col = self.social_u, self.social_v
        row_emb = user_emb.weight[row]
        col_emb = user_emb.weight[col]
        cat_emb = torch.cat([row_emb, col_emb], dim=1)
        out_layer1 = self.activate(self.linear_1(cat_emb))
        logit = self.linear_2(out_layer1)
        logit = logit.view(-1)
        eps = torch.rand(logit.shape).to(self.device)
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input) / 0.2
        mask_gate_input = torch.sigmoid(mask_gate_input) + self.edge_bias  # self.edge_bias
        weights = torch.ones_like(self.adj_matrix.values())
        weights[self.social_index] = mask_gate_input
        weights = weights.detach()
        masked_Graph = torch.sparse.FloatTensor(self.adj_matrix.indices(), self.adj_matrix.values()*weights, torch.Size(
            [self.num_user + self.num_item, self.num_user + self.num_item]))
        masked_Graph = masked_Graph.coalesce().to(self.device)
        return masked_Graph


    def forward(self, adj_matrix):
        '''
        LightGCN-S Encoders
        '''
        ego_emb = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_emb = [ego_emb]
        for _ in range(self.gcn_layer):
            tmp_emb = torch.sparse.mm(adj_matrix, all_emb[-1])
            all_emb.append(tmp_emb)
        all_emb = torch.stack(all_emb, dim=1)
        mean_emb = torch.mean(all_emb, dim=1)
        user_emb, item_emb = torch.split(mean_emb, [self.num_user, self.num_item])
        return user_emb, item_emb


    def getEmbedding(self, users, pos_items, neg_items):
        # all_users, all_items = self.forward()
        users_emb = self.user_emb[users]
        pos_emb = self.item_emb[pos_items]
        neg_emb = self.item_emb[neg_items]
        users_emb_ego = self.user_embeddings(users)
        pos_emb_ego = self.item_embeddings(pos_items)
        neg_emb_ego = self.item_embeddings(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


    def bpr_loss(self, users, pos_items, neg_items):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())
        reg_loss = 1/2 * (userEmb0.norm(2).pow(2) +
                    posEmb0.norm(2).pow(2) +
                    negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        auc = torch.mean((pos_scores > neg_scores).float())
        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        return auc, bpr_loss, reg_loss*self.l2_reg


    def hsic_graph(self, users, pos_items):
        ### user part ###
        users = torch.unique(users)
        items = torch.unique(pos_items)
        input_x = self.user_emb_old[users]
        input_y = self.user_emb[users]
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, self.batch_size)
        ### item part ###
        input_i = self.item_emb_old[items]
        input_j = self.item_emb[items]
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, self.batch_size)
        loss = loss_user + loss_item
        return loss


    def calculate_all_loss(self, users, pos_items, neg_items):
        # 1. learning denoised social graph
        self.masked_adj_matrix = self.graph_learner(self.user_embeddings)
        # 2. learning embeddings from lightgcn-s
        self.user_emb_old, self.item_emb_old = self.forward(self.adj_matrix)
        self.user_emb, self.item_emb = self.forward(self.masked_adj_matrix)
        # 3. Max mutual information
        auc, bpr_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items)
        # 4. Min mutual information
        ib_loss = self.hsic_graph(users, pos_items) * self.beta
        loss = bpr_loss + reg_loss + ib_loss
        return auc, bpr_loss, reg_loss, ib_loss, loss