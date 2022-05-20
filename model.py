import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.transformer import TransformerEncoder


class ClassEmbedding(nn.Module):
    def __init__(self, cfg, trainable=True):
        super(ClassEmbedding, self).__init__()
        idx2vocab = utils.load_files(cfg["DATASET"]["IDX2VOCAB"])
        self.n_token = len(idx2vocab)
        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]

        self.emb = nn.Embedding(self.n_token, self.word_emb_size)
        weight_init = utils.load_files(cfg["DATASET"]["GLOVE"]).astype(np.float32)
        weight_mat = torch.from_numpy(weight_init)
        self.emb.load_state_dict({"weight": weight_mat})

        if not trainable:
            self.emb.weight.requires_grad = False

    def forward(self, x):
        emb = self.emb(x)
        return emb


class AnswerSelector(nn.Module):
    def __init__(self, cfg):
        super(AnswerSelector, self).__init__()
        self.av2i = utils.load_files(cfg["DATASET"]["AVOCAB2IDX"])
        self.len_avocab = len(self.av2i)

        self.glove_cands = utils.load_files(cfg["DATASET"]["GLOVE_ANS_CAND"]).astype(
            np.float32
        )
        self.glove_cands = torch.from_numpy(self.glove_cands).cuda()

    def forward(self, inputs):
        similarity = torch.matmul(inputs, self.glove_cands.transpose(0, 1))
        pred = F.log_softmax(similarity, dim=1)
        return pred


class HypergraphTransformer(nn.Module):
    def __init__(self, cfg, args):
        super(HypergraphTransformer, self).__init__()

        self.cfg = cfg
        self.args = args
        self.n_hop = args.n_hop

        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.max_num_hqnode = cfg["MODEL"]["NUM_MAX_QNODE"]
        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.max_num_hknode = cfg["MODEL"]["NUM_MAX_KNODE_{}H".format(self.n_hop)]
        self.n_out = cfg["MODEL"]["NUM_OUT"]
        self.n_ans = cfg["MODEL"]["NUM_ANS"]
        self.abl_only_ga = args.abl_only_ga
        self.abl_only_sa = args.abl_only_sa

        if "pql" in args.data_name:
            self.i2e = ClassEmbedding(cfg, False)  # pql : small dataset
        else:
            self.i2e = ClassEmbedding(cfg)

        self.q2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hqnode, self.n_hidden
        )
        self.k2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hknode, self.n_hidden
        )

        if self.abl_only_sa != True:
            self.trans_k_with_q = self.get_network(self_type="kq")
            self.trans_q_with_k = self.get_network(self_type="qk")

        if self.abl_only_ga != True:
            self.trans_k_mem = self.get_network(self_type="k_mem", layers=3)
            self.trans_q_mem = self.get_network(self_type="q_mem", layers=3)

        self.dropout = nn.Dropout(p=self.cfg["MODEL"]["INP_DROPOUT"])
        self.out_dropout = 0.0

        if self.args.abl_ans_fc != True:
            self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
            self.proj2 = nn.Linear(self.n_hidden, self.n_out)
            self.ans_selector = AnswerSelector(cfg)
        else:
            self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
            self.proj2 = nn.Linear(self.n_hidden, self.n_ans)

    def get_network(self, self_type="", layers=-1):
        if self_type in ["kq", "k_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_K"]
        elif self_type in ["qk", "q_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_Q"]
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],
            layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
            fc_hid_coeff=self.cfg["MODEL"]["FC_HID_COEFF"],
        )

    def forward(self, batch):
        he_ques = batch[0]
        he_kg = batch[1]

        num_batch = he_ques.shape[0]
        num_he_ques = he_ques.shape[1]
        num_he_kg = he_kg.shape[1]

        he_ques = torch.reshape(self.i2e(he_ques), (num_batch, num_he_ques, -1))
        he_kg = torch.reshape(self.i2e(he_kg), (num_batch, num_he_kg, -1))

        he_ques = self.q2h(he_ques)
        he_kg = self.k2h(he_kg)

        he_ques = self.dropout(he_ques)
        he_kg = self.dropout(he_kg)

        he_ques = he_ques.permute(1, 0, 2)
        he_kg = he_kg.permute(1, 0, 2)

        if self.args.abl_only_ga == True:
            h_k_with_q = self.trans_k_with_q(he_kg, he_ques, he_ques)
            h_ks_sum = torch.sum(h_k_with_q, axis=0)
            h_q_with_k = self.trans_q_with_k(he_ques, he_kg, he_kg)
            h_qs_sum = torch.sum(h_q_with_k, axis=0)
            last_kq = torch.cat([h_ks_sum, h_qs_sum], dim=1)

        elif self.args.abl_only_sa == True:
            h_ks = self.trans_k_mem(he_kg)
            h_ks_sum = torch.sum(h_ks, axis=0)
            h_qs = self.trans_q_mem(he_ques)
            h_qs_sum = torch.sum(h_qs, axis=0)
            last_kq = torch.cat([h_ks_sum, h_qs_sum], dim=1)

        else:  # self.args.abl_only_ga == False and self.args.abl_only_sa == False:
            h_k_with_q = self.trans_k_with_q(he_kg, he_ques, he_ques)
            h_ks = self.trans_k_mem(h_k_with_q)
            h_ks_sum = torch.sum(h_ks, axis=0)

            h_q_with_k = self.trans_q_with_k(he_ques, he_kg, he_kg)
            h_qs = self.trans_q_mem(h_q_with_k)
            h_qs_sum = torch.sum(h_qs, axis=0)

            last_kq = torch.cat([h_ks_sum, h_qs_sum], dim=1)

        if self.args.abl_ans_fc != True:
            output = self.proj2(
                F.dropout(
                    F.relu(self.proj1(last_kq)),
                    p=self.out_dropout,
                    training=self.training,
                )
            )
            pred = self.ans_selector(output)
        else:
            output = self.proj2(
                F.dropout(
                    F.relu(self.proj1(last_kq)),
                    p=self.out_dropout,
                    training=self.training,
                )
            )
            pred = F.log_softmax(output, dim=1)
        return pred


class HypergraphTransformer_wohe(nn.Module):
    def __init__(self, cfg, args):
        super(HypergraphTransformer_wohe, self).__init__()

        self.cfg = cfg
        self.args = args
        self.n_hop = args.n_hop

        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_out = cfg["MODEL"]["NUM_OUT"]
        self.n_ans = cfg["MODEL"]["NUM_ANS"]
        self.max_num_hqnode = 1
        self.max_num_hknode = 1

        self.i2e = ClassEmbedding(cfg)
        self.q2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hqnode, self.n_hidden
        )
        self.k2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hknode, self.n_hidden
        )

        self.trans_k_with_q = self.get_network(self_type="kq")
        self.trans_q_with_k = self.get_network(self_type="qk")

        self.trans_k_mem = self.get_network(self_type="k_mem", layers=3)
        self.trans_q_mem = self.get_network(self_type="q_mem", layers=3)

        self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
        self.proj2 = nn.Linear(self.n_hidden, self.n_out)
        self.dropout = nn.Dropout(p=self.cfg["MODEL"]["INP_DROPOUT"])
        self.out_dropout = 0.0

        if self.args.abl_ans_fc != True:
            self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
            self.proj2 = nn.Linear(self.n_hidden, self.n_out)
            self.ans_selector = AnswerSelector(cfg)
        else:
            self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
            self.proj2 = nn.Linear(self.n_hidden, self.n_ans)

    def get_network(self, self_type="", layers=-1):
        if self_type in ["kq", "k_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_K"]
        elif self_type in ["qk", "q_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_Q"]
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],
            layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
        )

    def forward(self, batch):
        he_ques = batch[0]
        he_kg = batch[1]

        num_batch = he_ques.shape[0]
        num_he_ques = he_ques.shape[1]
        num_he_kg = he_kg.shape[1]

        he_ques = torch.reshape(self.i2e(he_ques), (num_batch, num_he_ques, -1))
        he_kg = torch.reshape(self.i2e(he_kg), (num_batch, num_he_kg, -1))

        he_ques = self.q2h(he_ques)
        he_kg = self.k2h(he_kg)

        he_ques = self.dropout(he_ques)
        he_kg = self.dropout(he_kg)

        he_ques = he_ques.permute(1, 0, 2)
        he_kg = he_kg.permute(1, 0, 2)

        h_k_with_q = self.trans_k_with_q(he_kg, he_ques, he_ques)
        h_ks = self.trans_k_mem(h_k_with_q)
        h_ks_sum = torch.sum(h_ks, axis=0)

        h_q_with_k = self.trans_q_with_k(he_ques, he_kg, he_kg)
        h_qs = self.trans_q_mem(h_q_with_k)
        h_qs_sum = torch.sum(h_qs, axis=0)

        last_kq = torch.cat([h_ks_sum, h_qs_sum], dim=1)

        if self.args.abl_ans_fc != True:
            output = self.proj2(
                F.dropout(
                    F.relu(self.proj1(last_kq)),
                    p=self.out_dropout,
                    training=self.training,
                )
            )
            pred = self.ans_selector(output)
        else:
            output = self.proj2(
                F.dropout(
                    F.relu(self.proj1(last_kq)),
                    p=self.out_dropout,
                    training=self.training,
                )
            )
            pred = F.log_softmax(output, dim=1)
        return pred


class HypergraphTransformer_qsetkhe(nn.Module):
    def __init__(self, cfg, args):
        super(HypergraphTransformer_qsetkhe, self).__init__()

        self.cfg = cfg
        self.args = args
        self.n_hop = args.n_hop

        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_out = cfg["MODEL"]["NUM_OUT"]
        self.max_num_hqnode = 1
        self.max_num_hknode = cfg["MODEL"]["NUM_MAX_KNODE_{}H".format(self.n_hop)]

        self.i2e = ClassEmbedding(cfg)
        self.q2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hqnode, self.n_hidden
        )
        self.k2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hknode, self.n_hidden
        )

        self.trans_k_with_q = self.get_network(self_type="kq")
        self.trans_q_with_k = self.get_network(self_type="qk")

        self.trans_k_mem = self.get_network(self_type="k_mem", layers=3)
        self.trans_q_mem = self.get_network(self_type="q_mem", layers=3)

        self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
        self.proj2 = nn.Linear(self.n_hidden, self.n_out)
        self.dropout = nn.Dropout(p=self.cfg["MODEL"]["INP_DROPOUT"])
        self.out_dropout = 0.0

        self.ans_selector = AnswerSelector(cfg)

    def get_network(self, self_type="", layers=-1):
        if self_type in ["kq", "k_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_K"]
        elif self_type in ["qk", "q_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_Q"]
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],
            layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
        )

    def forward(self, batch):
        he_ques = batch[0]
        he_kg = batch[1]

        num_batch = he_ques.shape[0]
        num_he_ques = he_ques.shape[1]
        num_he_kg = he_kg.shape[1]

        he_ques = torch.reshape(self.i2e(he_ques), (num_batch, num_he_ques, -1))
        he_kg = torch.reshape(self.i2e(he_kg), (num_batch, num_he_kg, -1))

        he_ques = self.q2h(he_ques)
        he_kg = self.k2h(he_kg)

        he_ques = self.dropout(he_ques)
        he_kg = self.dropout(he_kg)

        he_ques = he_ques.permute(1, 0, 2)
        he_kg = he_kg.permute(1, 0, 2)

        h_k_with_q = self.trans_k_with_q(he_kg, he_ques, he_ques)
        h_ks = self.trans_k_mem(h_k_with_q)
        h_ks_sum = torch.sum(h_ks, axis=0)

        h_q_with_k = self.trans_q_with_k(he_ques, he_kg, he_kg)
        h_qs = self.trans_q_mem(h_q_with_k)
        h_qs_sum = torch.sum(h_qs, axis=0)

        last_kq = torch.cat([h_ks_sum, h_qs_sum], dim=1)

        output = self.proj2(
            F.dropout(
                F.relu(self.proj1(last_kq)), p=self.out_dropout, training=self.training
            )
        )
        pred = self.ans_selector(output)
        return pred


class HypergraphTransformer_qhekset(nn.Module):
    def __init__(self, cfg, args):
        super(HypergraphTransformer_qhekset, self).__init__()

        self.cfg = cfg
        self.args = args
        self.n_hop = args.n_hop

        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_out = cfg["MODEL"]["NUM_OUT"]
        self.max_num_hknode = 1
        self.max_num_hqnode = cfg["MODEL"]["NUM_MAX_QNODE"]

        self.i2e = ClassEmbedding(cfg)
        self.q2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hqnode, self.n_hidden
        )
        self.k2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hknode, self.n_hidden
        )

        self.trans_k_with_q = self.get_network(self_type="kq")
        self.trans_q_with_k = self.get_network(self_type="qk")

        self.trans_k_mem = self.get_network(self_type="k_mem", layers=3)
        self.trans_q_mem = self.get_network(self_type="q_mem", layers=3)

        self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
        self.proj2 = nn.Linear(self.n_hidden, self.n_out)
        self.dropout = nn.Dropout(p=self.cfg["MODEL"]["INP_DROPOUT"])
        self.out_dropout = 0.0

        self.ans_selector = AnswerSelector(cfg)

    def get_network(self, self_type="", layers=-1):
        if self_type in ["kq", "k_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_K"]
        elif self_type in ["qk", "q_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_Q"]
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],
            layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
        )

    def forward(self, batch):
        he_ques = batch[0]
        he_kg = batch[1]

        num_batch = he_ques.shape[0]
        num_he_ques = he_ques.shape[1]
        num_he_kg = he_kg.shape[1]

        he_ques = torch.reshape(self.i2e(he_ques), (num_batch, num_he_ques, -1))
        he_kg = torch.reshape(self.i2e(he_kg), (num_batch, num_he_kg, -1))

        he_ques = self.q2h(he_ques)
        he_kg = self.k2h(he_kg)

        he_ques = self.dropout(he_ques)
        he_kg = self.dropout(he_kg)

        he_ques = he_ques.permute(1, 0, 2)
        he_kg = he_kg.permute(1, 0, 2)

        h_k_with_q = self.trans_k_with_q(he_kg, he_ques, he_ques)
        h_ks = self.trans_k_mem(h_k_with_q)
        h_ks_sum = torch.sum(h_ks, axis=0)

        h_q_with_k = self.trans_q_with_k(he_ques, he_kg, he_kg)
        h_qs = self.trans_q_mem(h_q_with_k)
        h_qs_sum = torch.sum(h_qs, axis=0)

        last_kq = torch.cat([h_ks_sum, h_qs_sum], dim=1)

        output = self.proj2(
            F.dropout(
                F.relu(self.proj1(last_kq)), p=self.out_dropout, training=self.training
            )
        )
        pred = self.ans_selector(output)
        return pred


class HAN(nn.Module):
    def __init__(self, cfg, args):
        super(HAN, self).__init__()

        self.cfg = cfg

        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_head = cfg["MODEL"]["NUM_HEAD"]
        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.n_hop = args.n_hop
        self.n_out = cfg["MODEL"]["NUM_OUT"]

        self.max_num_hk = cfg["MODEL"]["NUM_MAX_HK_{}H".format(self.n_hop)]
        self.max_num_hknode = cfg["MODEL"]["NUM_MAX_KNODE_{}H".format(self.n_hop)]
        self.max_num_hqnode = cfg["MODEL"]["NUM_MAX_QNODE"]

        self.i2e = ClassEmbedding(cfg)
        self.q2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hqnode, self.n_hidden
        )
        self.k2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hknode, self.n_hidden
        )

        self.h2att = torch.nn.Linear(self.n_hidden, self.n_head)
        self.softmax_att = torch.nn.Softmax(dim=2)
        self.fc_out = torch.nn.Linear(self.n_hidden * self.n_head, self.n_out)
        self.dropout = nn.Dropout(p=self.cfg["MODEL"]["INP_DROPOUT"])

        self.ans_selector = AnswerSelector(cfg)

    def multihead_att(self, he_ques, he_src):
        num_batch = he_ques.shape[0]
        num_he_ques = he_ques.shape[1]
        num_he_src = he_src.shape[1]

        he_ques = torch.reshape(self.i2e(he_ques), (num_batch, num_he_ques, -1))
        he_src = torch.reshape(self.i2e(he_src), (num_batch, num_he_src, -1))

        he_ques = self.q2h(he_ques)
        he_src = self.k2h(he_src)

        he_ques = self.dropout(he_ques)
        he_src = self.dropout(he_src)

        he_ques = he_ques.permute(0, 2, 1)
        he_src = he_src.permute(0, 2, 1)

        he_ques_selfatt = he_ques.unsqueeze(3)
        he_src_selfatt = he_src.unsqueeze(2)

        self_mul = torch.matmul(he_ques_selfatt, he_src_selfatt)
        self_mul = self_mul.permute(0, 2, 3, 1)

        att_map = self.h2att(self_mul)
        att_map = att_map.permute(0, 3, 1, 2)

        att_map = torch.reshape(att_map, (-1, self.n_head, num_he_ques * num_he_src))
        att_map = self.softmax_att(att_map)
        att_map = torch.reshape(att_map, (-1, self.n_head, num_he_ques, num_he_src))
        he_ques = he_ques.unsqueeze(2)
        he_src = he_src.unsqueeze(3)

        for i in range(self.n_head):
            att_g = att_map[:, i : i + 1, :, :]
            att_g_t = att_g.repeat([1, self.n_hidden, 1, 1])
            att_out = torch.matmul(he_ques, att_g_t)
            att_out = torch.matmul(att_out, he_src)
            att_out = att_out.squeeze(-1)
            att_out_sq = att_out.squeeze(-1)

            if i == 0:
                output = att_out_sq
            else:
                output = torch.cat((output, att_out_sq), dim=1)

        output = self.fc_out(output)
        pred = self.ans_selector(output)
        return pred, att_map

    def forward(self, batch):
        he_ques = batch[0]
        he_kg = batch[1]

        pred, att_map = self.multihead_att(he_ques, he_kg)
        return pred


class BAN(nn.Module):
    def __init__(self, cfg, args):
        super(BAN, self).__init__()

        self.cfg = cfg

        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_head = cfg["MODEL"]["NUM_HEAD"]
        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.n_hop = args.n_hop
        self.n_out = cfg["MODEL"]["NUM_OUT"]

        self.max_num_hk = cfg["MODEL"]["NUM_MAX_HK_{}H".format(self.n_hop)]
        self.max_num_hknode = 1
        self.max_num_hqnode = 1

        self.i2e = ClassEmbedding(cfg)
        self.q2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hqnode, self.n_hidden
        )
        self.k2h = torch.nn.Linear(
            self.word_emb_size * self.max_num_hknode, self.n_hidden
        )

        self.h2att = torch.nn.Linear(self.n_hidden, self.n_head)
        self.softmax_att = torch.nn.Softmax(dim=2)
        self.fc_out = torch.nn.Linear(self.n_hidden * self.n_head, self.n_out)
        self.dropout = nn.Dropout(p=self.cfg["MODEL"]["INP_DROPOUT"])

        self.ans_selector = AnswerSelector(cfg)

    def multihead_att(self, he_ques, he_src, q2h, s2h):
        num_batch = he_ques.shape[0]
        num_he_ques = he_ques.shape[1]
        num_he_src = he_src.shape[1]

        he_ques = torch.reshape(self.i2e(he_ques), (num_batch, num_he_ques, -1))
        he_src = torch.reshape(self.i2e(he_src), (num_batch, num_he_src, -1))

        he_ques = q2h(he_ques)
        he_src = s2h(he_src)

        he_ques = self.dropout(he_ques)
        he_src = self.dropout(he_src)

        he_ques = he_ques.permute(0, 2, 1)
        he_src = he_src.permute(0, 2, 1)

        he_ques_selfatt = he_ques.unsqueeze(3)
        he_src_selfatt = he_src.unsqueeze(2)

        self_mul = torch.matmul(he_ques_selfatt, he_src_selfatt)
        self_mul = self_mul.permute(0, 2, 3, 1)

        att_map = self.h2att(self_mul)
        att_map = att_map.permute(0, 3, 1, 2)

        att_map = torch.reshape(att_map, (-1, self.n_head, num_he_ques * num_he_src))
        att_map = self.softmax_att(att_map)
        att_map = torch.reshape(att_map, (-1, self.n_head, num_he_ques, num_he_src))
        he_ques = he_ques.unsqueeze(2)
        he_src = he_src.unsqueeze(3)

        for i in range(self.n_head):
            att_g = att_map[:, i : i + 1, :, :]
            att_g_t = att_g.repeat([1, self.n_hidden, 1, 1])
            att_out = torch.matmul(he_ques, att_g_t)
            att_out = torch.matmul(att_out, he_src)
            att_out = att_out.squeeze(-1)
            att_out_sq = att_out.squeeze(-1)

            if i == 0:
                output = att_out_sq
            else:
                output = torch.cat((output, att_out_sq), dim=1)

        output = self.fc_out(output)
        pred = self.ans_selector(output)
        return pred, att_map

    def forward(self, batch):
        he_ques = batch[0]
        he_kg = batch[1]

        pred, att_map = self.multihead_att(he_ques, he_kg, self.q2h, self.k2h)
        return pred


class GGNN(nn.Module):
    """
    Reimplementation of Gated Graph Sequence Neural Networks (GGNN) by Kaihua Tang
    Implementation based on https://arxiv.org/abs/1511.05493
    """

    def __init__(self, cfg, args, n_node):
        super(GGNN, self).__init__()
        self.n_input = cfg["MODEL"]["NUM_WORD_EMB"]
        self.annotation_dim = cfg["MODEL"]["NUM_ANNO"]
        self.hidden_dim = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_edge = cfg["MODEL"]["NUM_EDGE"]
        self.n_out = cfg["MODEL"]["NUM_OUT"]
        self.n_steps = cfg["MODEL"]["NUM_STEP"]

        self.max_num_kg = n_node
        self.max_num_q = cfg["MODEL"]["NUM_MAX_Q"]

        self.i2e = ClassEmbedding(cfg)
        self.fc_qenc = nn.Linear(self.n_input + self.annotation_dim, self.hidden_dim)
        self.fc_kenc = nn.Linear(self.n_input + self.annotation_dim, self.hidden_dim)
        self.fc_in = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)
        self.fc_out = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)

        self.gated_update_kg = GatedPropagation(
            self.hidden_dim, self.max_num_kg, self.n_edge
        )
        self.graph_aggregate_kg = GraphFeature(
            self.hidden_dim, self.max_num_kg, self.n_edge, self.annotation_dim
        )
        self.gated_update_ques = GatedPropagation(
            self.hidden_dim, self.max_num_q, self.n_edge
        )
        self.graph_aggregate_ques = GraphFeature(
            self.hidden_dim, self.max_num_q, self.n_edge, self.annotation_dim
        )

        self.fc_output = nn.Linear(self.hidden_dim * 2, self.n_out)

        self.ans_selector = AnswerSelector(cfg)

    def forward(self, batch):
        """
        batch: adj_matrix, annotation, entity_rep, answer
        init state x: [batch_size, num_node, hidden_size]
        annoatation a: [batch_size, num_node, 1]
        adj matrix m: [batch_size, num_node, num_node * n_edge_types * 2]
        output out: [batch_size, n_label]
        """

        ques = batch[0]
        adjmat_ques = batch[1]
        ques_anno = batch[2]
        kg = batch[3]
        adjmat_kg = batch[4]
        kg_anno = batch[5]

        kg = self.i2e(kg)
        ques = self.i2e(ques)

        kg = torch.cat((kg, kg_anno), 2)
        ques = torch.cat((ques, ques_anno), 2)

        kg = self.fc_kenc(kg)
        ques = self.fc_qenc(ques)

        for i in range(self.n_steps):
            in_states = self.fc_in(kg)
            out_states = self.fc_out(kg)
            in_states = (
                in_states.view(-1, self.max_num_kg, self.hidden_dim, self.n_edge)
                .transpose(2, 3)
                .transpose(1, 2)
                .contiguous()
            )
            in_states = in_states.view(
                -1, self.max_num_kg * self.n_edge, self.hidden_dim
            )
            out_states = (
                out_states.view(-1, self.max_num_kg, self.hidden_dim, self.n_edge)
                .transpose(2, 3)
                .transpose(1, 2)
                .contiguous()
            )
            out_states = out_states.view(
                -1, self.max_num_kg * self.n_edge, self.hidden_dim
            )
            kg = self.gated_update_kg(in_states, out_states, kg, adjmat_kg)

        for i in range(self.n_steps):
            in_states = self.fc_in(ques)
            out_states = self.fc_out(ques)
            in_states = (
                in_states.view(-1, self.max_num_q, self.hidden_dim, self.n_edge)
                .transpose(2, 3)
                .transpose(1, 2)
                .contiguous()
            )
            in_states = in_states.view(
                -1, self.max_num_q * self.n_edge, self.hidden_dim
            )
            out_states = (
                out_states.view(-1, self.max_num_q, self.hidden_dim, self.n_edge)
                .transpose(2, 3)
                .transpose(1, 2)
                .contiguous()
            )
            out_states = out_states.view(
                -1, self.max_num_q * self.n_edge, self.hidden_dim
            )
            ques = self.gated_update_ques(in_states, out_states, ques, adjmat_ques)

        kg_out = self.graph_aggregate_kg(torch.cat((kg, kg_anno), 2))
        ques_out = self.graph_aggregate_ques(torch.cat((ques, ques_anno), 2))

        output = torch.cat((kg_out, ques_out), axis=1)
        output = self.fc_output(output)

        pred = self.ans_selector(output)
        return pred


class GraphFeature(nn.Module):
    def __init__(self, hidden_dim, n_node, n_edge, n_anno):
        super(GraphFeature, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_anno = n_anno

        self.fc_i = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.fc_j = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_sigm = self.sigmoid(self.fc_i(x))
        x_tanh = self.tanh(self.fc_j(x))
        x_new = (x_sigm * x_tanh).sum(1)

        return self.tanh(x_new)


class GatedPropagation(nn.Module):
    def __init__(self, hidden_dim, n_node, n_edge):
        super(GatedPropagation, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge

        self.gate_r = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.gate_z = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.trans = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_in, x_out, x_curt, matrix):
        matrix_in = matrix[:, :, : self.n_node * self.n_edge]
        matrix_out = matrix[:, :, self.n_node * self.n_edge :]

        a_in = torch.bmm(matrix_in.float(), x_in)
        a_out = torch.bmm(matrix_out.float(), x_out)
        a = torch.cat((a_in, a_out, x_curt), 2)

        z = self.sigmoid(self.gate_z(a))
        r = self.sigmoid(self.gate_r(a))

        joint_input = torch.cat((a_in, a_out, r * x_curt), 2)
        h_hat = self.tanh(self.trans(joint_input))
        output = (1 - z) * x_curt + z * h_hat
        return output


class GCN(torch.nn.Module):
    def __init__(self, cfg, arg):
        super(GCN, self).__init__()

        self.n_input = cfg["MODEL"]["NUM_WORD_EMB"]
        self.hidden_dim = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_out = cfg["MODEL"]["NUM_OUT"]
        self.i2e = ClassEmbedding(cfg)

        self.q_gcn1 = DenseGCNConv(self.n_input, self.hidden_dim)
        self.q_gcn2 = DenseGCNConv(self.hidden_dim, self.hidden_dim)
        self.kg_gcn1 = DenseGCNConv(self.n_input, self.hidden_dim)
        self.kg_gcn2 = DenseGCNConv(self.hidden_dim, self.hidden_dim)

        self.fc_output = nn.Linear(self.hidden_dim * 2, self.n_out)
        self.ans_selector = AnswerSelector(cfg)

    def forward(self, batch):
        ques_idxs = batch[0]
        ques_adj = batch[1]
        kg_idxs = batch[2]
        kg_adj = batch[3]

        ques_emb = self.i2e(ques_idxs)
        kg_emb = self.i2e(kg_idxs)

        ques_emb = self.q_gcn1(ques_emb, ques_adj)
        ques_emb = self.q_gcn2(ques_emb, ques_adj)
        ques_emb = torch.sum(ques_emb, axis=1)

        kg_emb = self.q_gcn1(kg_emb, kg_adj)
        kg_emb = self.q_gcn2(kg_emb, kg_adj)
        kg_emb = torch.sum(kg_emb, axis=1)

        last_kg = torch.cat([kg_emb, ques_emb], dim=1)
        output = self.fc_output(last_kg)
        pred = self.ans_selector(output)

        return pred


class DenseGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.register_parameter("gcn_weight", self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
            self.register_parameter("gcn_bias", self.bias)

        self.reset_parameters()

    def reset_parameters(self):
        utils.glorot(self.weight)
        utils.zeros(self.bias)

    def forward(self, x, adj, mask=None, add_loop=True):
        x = x.float()
        adj = adj.float()

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class MemNet(nn.Module):
    def __init__(self, cfg, args):
        super(MemNet, self).__init__()

        self.cfg = cfg
        self.args = args

        self.n_steps = cfg["MODEL"]["NUM_STEP"]
        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.dropout = nn.Dropout(p=self.cfg["MODEL"]["DROPOUT"])

        self.i2e_ab = ClassEmbedding(cfg)
        if cfg["MODEL"]["SHARE_FLAG"] == True:
            self.i2e_c = self.i2e_ab
        else:
            self.i2e_c = ClassEmbedding(cfg)

        self.ans_selector = AnswerSelector(cfg)

    def forward(self, batch):
        q = batch[0]
        x = batch[1]

        bs = x.size(0)
        story_len = x.size(1)
        s_sent_len = x.size(2)

        x = x.view(bs * story_len, -1)
        u = self.dropout(self.i2e_ab(q))
        u = torch.sum(torch.sum(u, 1), 1)

        for k in range(self.n_steps):
            m = self.dropout(self.i2e_ab(x))
            m = m.view(bs, story_len, s_sent_len, -1)
            m = torch.sum(m, 2)

            c = self.dropout(self.i2e_c(x))
            c = c.view(bs, story_len, s_sent_len, -1)
            c = torch.sum(c, 2)

            p = torch.bmm(m, u.unsqueeze(2))
            p = torch.bmm(m, u.unsqueeze(2)).squeeze(2)
            p = F.softmax(p, -1).unsqueeze(1)

            o = torch.bmm(p, c).squeeze(1)
            u = o + u

        pred = self.ans_selector(u)
        return pred
