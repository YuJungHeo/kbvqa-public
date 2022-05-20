import os
import json
import time
import torch
import random
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from nltk.util import ngrams
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def sampling(prob):
    return np.random.choice(len(prob), 1, p=prob)


def load_files(path):
    if path.rsplit(".", 2)[-1] == "json":
        with open(path, "r") as f:
            data = json.load(f)

    elif path.rsplit(".", 2)[-1] in ["pkl", "pickle"]:
        with open(path, "rb") as f:
            data = pickle.load(f)
    return data


def create_adjacency_matrix_qgraph(n_nodes, n_edge_types, entity_dict, ques):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    q_len = len(ques)

    for i in range(q_len - 1):
        src_idx = entity_dict[ques[i]]
        tgt_idx = entity_dict[ques[i + 1]]
        e_type = 0
        a[tgt_idx - 1][(e_type - 1) * n_nodes + src_idx - 1] = 1
        a[src_idx - 1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] = 1

    return a


def create_adjacency_matrix_kgraph(n_nodes, n_edge_types, entity_dict, kgs):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for kg in kgs:
        k_len = len(kg)
        for i in range(k_len - 1):
            src_idx = kg[i]
            if src_idx < 20:
                continue
            else:
                src_idx = entity_dict[src_idx]

                temp_idx = kg[i + 1]
                if temp_idx < 20:
                    e_type = temp_idx
                    tgt_idx = entity_dict[kg[i + 2]]
                else:
                    e_type = 0
                    tgt_idx = entity_dict[temp_idx]

                a[tgt_idx - 1][(e_type - 1) * n_nodes + src_idx - 1] = 1
                a[src_idx - 1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] = 1

    return a


def create_adjacency_matrix_flattengraph(n_nodes, entity_dict, kgs, max_num_nodes=-1):
    if max_num_nodes != -1:
        a = np.zeros([max_num_nodes, max_num_nodes])
    else:
        a = np.zeros([n_nodes, n_nodes])

    for kg in kgs:
        k_len = len(kg)
        for i in range(k_len - 1):
            src_idx = kg[i]
            src_idx = entity_dict[src_idx]
            tgt_idx = entity_dict[kg[i + 1]]
            a[src_idx, tgt_idx] = 1
    return a


def he_sampling(adj, num_steps, max_num_he, eps_prob=0.001):
    adj = (adj > 0.0).astype(float)
    n_nodes = adj.shape[0]

    num_outedges = np.sum(adj, axis=1) + 0.5
    init_prob = num_outedges / np.sum(num_outedges, keepdims=True)

    adj = adj + eps_prob
    row_sum = np.sum(adj, axis=1, keepdims=True)
    adj = adj / row_sum

    start_node = np.random.choice(
        n_nodes, max_num_he * 2, p=init_prob
    )  # can assign the start node as qids

    HEs = [[n] for n in start_node]
    for k in range(num_steps - 1):
        [HE.append(sampling(adj[HE[-1]])[0]) for HE in HEs]

    unique_HEs = []
    for HE in HEs:
        unique_HEs.append(list(np.unique(HE)))
    unique_HEs.append([0])
    unique_HEs = list(np.unique(unique_HEs))

    num_HE = min(max_num_he, len(unique_HEs))
    HEs = unique_HEs[:num_HE]

    return HEs


def load_PQnPQL_data(cfg, args):
    data = load_files(cfg["DATASET"]["PROC_DATA"])

    ques = data["ques"]
    ans = data["ans"]

    # used for only evaluation
    path = data["path"]
    aset = data["aset"]

    n_hop_cfg = "KG_%shop" % (args.n_hop)
    kghop = load_files(cfg["DATASET"][n_hop_cfg])

    # split randomly, codes from https://github.com/zmtkeke/IRN/blob/master/train.py
    (
        trainQ,
        testQ,
        trainA,
        testA,
        trainP,
        testP,
        trainAset,
        testAset,
        trainKB,
        testKB,
    ) = train_test_split(
        ques, ans, path, aset, kghop, test_size=0.1, random_state=args.split_seed
    )
    (
        trainQ,
        validQ,
        trainA,
        validA,
        trainP,
        validP,
        trainAset,
        validAset,
        trainKB,
        validKB,
    ) = train_test_split(
        trainQ, trainA, trainP, trainAset, trainKB, test_size=0.11, random_state=0
    )

    train = {}
    train["ques"] = trainQ
    train["ans"] = trainA
    train["path"] = trainP
    train["aset"] = trainAset
    train["KB"] = trainKB

    valid = {}
    valid["ques"] = validQ
    valid["ans"] = validA
    valid["path"] = validP
    valid["aset"] = validAset
    valid["KB"] = validKB

    test = {}
    test["ques"] = testQ
    test["ans"] = testA
    test["path"] = testP
    test["aset"] = testAset
    test["KB"] = testKB

    print(
        "num of data split: %s for train, %s for val, %s for test"
        % (trainQ.shape[0], validQ.shape[0], testQ.shape[0])
    )
    return train, valid, test


class PQnPQL(Dataset):
    def __init__(self, cfg, args, data):
        self.cfg = cfg
        self.args = args

        self.v2i = load_files(cfg["DATASET"]["VOCAB2IDX"])
        self.i2v = load_files(cfg["DATASET"]["IDX2VOCAB"])
        self.len_vocab = len(self.v2i)
        self.av2i = load_files(cfg["DATASET"]["AVOCAB2IDX"])

        self.n_hop = args.n_hop
        self.n_edge = cfg["MODEL"]["NUM_EDGE"]
        self.max_num_q = cfg["MODEL"]["NUM_MAX_Q"]
        self.max_num_aset = cfg["MODEL"]["NUM_MAX_ASET"]

        if "ht" in self.args.model_name:
            self.max_num_hk = cfg["MODEL"]["NUM_MAX_HK_{}H".format(self.n_hop)]
            self.max_num_hknode = cfg["MODEL"]["NUM_MAX_KNODE_{}H".format(self.n_hop)]
            self.max_num_hqnode = cfg["MODEL"]["NUM_MAX_QNODE"]

        self.glove = load_files(cfg["DATASET"]["GLOVE"]).astype(np.float32)
        self.data = data

    def __len__(self):
        return self.data["ques"].shape[0]

    def __getitem__(self, idx):
        if self.args.model_name == "ht":
            ques = np.array(self.data["ques"][idx], dtype=np.int32)
            ques_he = np.array(list(ngrams(ques, self.max_num_hqnode)))
            ques_out = np.zeros((self.max_num_q, self.max_num_hqnode))
            ques_out[: ques_he.shape[0]] = ques_he
            ques_out = torch.from_numpy(ques_out).long()

            kgs = self.data["KB"][idx]

            if len(kgs) > self.max_num_hk:
                random.shuffle(kgs)
            numiter_kg = min(len(kgs), self.max_num_hk)
            kg_out = np.zeros((self.max_num_hk, self.max_num_hknode))

            for i in range(numiter_kg):
                kg = np.array(kgs[i])
                kg_out[i, : kg.shape[0]] = kg
            kg_out = torch.from_numpy(kg_out).long()

            ans = self.data["ans"][idx]
            aset = self.data["aset"][idx]
            aset_out = np.ones((self.max_num_aset)) * -1
            for i, aidx in enumerate(aset):
                aset_out[i] = aidx
            return ques_out, kg_out, aset_out, ans

        elif self.args.model_name == "ht_abl_wohe":
            ques = np.array(self.data["ques"][idx], dtype=np.int32)
            ques_out = np.zeros((self.max_num_q))
            len_ques = min(len(ques), self.max_num_q)
            ques_out[:len_ques] = ques[:len_ques]
            ques_out = torch.from_numpy(ques_out).long()

            kgs = self.data["KB"][idx]
            kgs = list(itertools.chain(*kgs))

            if len(kgs) > self.max_num_hk:
                random.shuffle(kgs)
            numiter_kg = min(len(kgs), self.max_num_hk)
            kg_out = np.zeros((self.max_num_hk))

            for i in range(numiter_kg):
                kg_out[i] = kgs[i]
            kg_out = torch.from_numpy(kg_out).long()

            ans = self.data["ans"][idx]
            aset = self.data["aset"][idx]
            aset_out = np.ones((self.max_num_aset)) * -1
            for i, aidx in enumerate(aset):
                aset_out[i] = aidx
            return ques_out, kg_out, aset_out, ans


def load_FVQA_data(cfg, args):
    data = load_files(cfg["DATASET"]["PROC_DATA"])

    ques = data["ques"]
    ans = data["ans"]
    image_fns = data["image_fn"]
    kb = load_files(cfg["DATASET"]["KB"])

    split_num = args.data_name[-1]
    split_train_fn = cfg["DATASET"]["SPLIT_DIR"] + "train_list_%s.txt" % (split_num)
    split_test_fn = cfg["DATASET"]["SPLIT_DIR"] + "test_list_%s.txt" % (split_num)

    if os.path.isfile(split_train_fn):
        with open(split_train_fn) as f:
            lines = f.readlines()

    train_fns = []
    for line in lines:
        line = line.strip()
        train_fns.append(line)

    if os.path.isfile(split_test_fn):
        with open(split_test_fn) as f:
            lines = f.readlines()

    test_fns = []
    for line in lines:
        line = line.strip()
        test_fns.append(line)

    trainQ = []
    trainA = []
    trainKB = []
    testQ = []
    testA = []
    testKB = []

    for i, image_fn in enumerate(image_fns):
        if image_fn in train_fns:
            trainQ.append(ques[i])
            trainA.append(ans[i])
            trainKB.append(kb[i])

        elif image_fn in test_fns:
            testQ.append(ques[i])
            testA.append(ans[i])
            testKB.append(kb[i])

        else:
            print(image_fn)

    train = {}
    train["ques"] = trainQ
    train["ans"] = trainA
    train["KB"] = trainKB

    test = {}
    test["ques"] = testQ
    test["ans"] = testA
    test["KB"] = testKB

    print("num of data split: %s for train, %s for test" % (len(trainQ), len(testQ)))
    return train, test


class FVQA(Dataset):
    def __init__(self, cfg, args, data):
        self.cfg = cfg
        self.args = args

        self.v2i = load_files(cfg["DATASET"]["VOCAB2IDX"])
        self.i2v = load_files(cfg["DATASET"]["IDX2VOCAB"])
        self.len_vocab = len(self.v2i)
        self.av2i = load_files(cfg["DATASET"]["AVOCAB2IDX"])

        self.n_hop = args.n_hop
        self.max_num_q = cfg["MODEL"]["NUM_MAX_Q"]

        if "ht" in self.args.model_name:
            self.max_num_hk = cfg["MODEL"]["NUM_MAX_HK_{}H".format(self.n_hop)]
            self.max_num_hknode = cfg["MODEL"]["NUM_MAX_KNODE_{}H".format(self.n_hop)]
            self.max_num_hqnode = cfg["MODEL"]["NUM_MAX_QNODE"]

        self.glove = load_files(cfg["DATASET"]["GLOVE"]).astype(np.float32)
        self.data = data

    def __len__(self):
        return len(self.data["ques"])

    def __getitem__(self, idx):
        if self.args.model_name == "ht":
            ques = np.array(self.data["ques"][idx], dtype=np.int32)
            ques_he = np.array(list(ngrams(ques, self.max_num_hqnode)))
            ques_out = np.zeros((self.max_num_q, self.max_num_hqnode))
            ques_out[: ques_he.shape[0]] = ques_he
            ques_out = torch.from_numpy(ques_out).long()

            kgs = self.data["KB"][idx]
            if len(kgs) > self.max_num_hk:
                random.shuffle(kgs)
            numiter_kg = min(len(kgs), self.max_num_hk)
            kg_out = np.zeros((self.max_num_hk, self.max_num_hknode))

            for i in range(numiter_kg):
                kg = np.array(kgs[i])
                kg_out[i, : kg.shape[0]] = kg
            kg_out = torch.from_numpy(kg_out).long()

            ans = self.data["ans"][idx]
            return ques_out, kg_out, ans


class KVQA(Dataset):
    def __init__(self, cfg, args, mode, task_idx=-1):
        self.cfg = cfg
        self.args = args

        self.v2i = load_files(cfg["DATASET"]["VOCAB2IDX"])
        self.i2v = load_files(cfg["DATASET"]["IDX2VOCAB"])
        self.len_vocab = len(self.v2i)
        self.av2i = load_files(cfg["DATASET"]["AVOCAB2IDX"])

        self.n_hop = args.n_hop
        self.n_edge = cfg["MODEL"]["NUM_EDGE"]
        self.max_num_q = cfg["MODEL"]["NUM_MAX_Q"]

        if self.args.model_name == "ht" or self.args.model_name == "memnet":
            self.max_num_hk = cfg["MODEL"]["NUM_MAX_HK_{}H".format(self.n_hop)]
            self.max_num_hknode = cfg["MODEL"]["NUM_MAX_KNODE_{}H".format(self.n_hop)]
            self.max_num_hqnode = cfg["MODEL"]["NUM_MAX_QNODE"]

        elif self.args.model_name == "ban" or self.args.model_name == "ht_abl_wohe":
            self.max_num_hk = cfg["MODEL"]["NUM_MAX_HK_{}H".format(self.n_hop)]

        elif self.args.model_name == "han":
            self.max_num_hk = cfg["MODEL"]["NUM_MAX_HK_{}H".format(self.n_hop)]
            self.max_num_hknode = cfg["MODEL"]["NUM_MAX_KNODE_{}H".format(self.n_hop)]
            self.max_num_hqnode = cfg["MODEL"]["NUM_MAX_QNODE"]

        self.img_idx = []  # for visualization

        if self.args.selected == True:
            self.data = self.load_data_selected()
        else:
            self.data = self.load_data(mode, task_idx)

        self.oshot_ans_idxs, self.zshot_ans_idxs = self.create_ans_mask()
        self.glove = load_files(cfg["DATASET"]["GLOVE"]).astype(np.float32)
        self.max_n_node = max(self.n_node)

    def __len__(self):
        return len(self.data["ques"])

    def __getitem__(self, idx):
        if self.args.model_name == "ggnn":
            ques = self.data["ques"][idx]
            ques_out = np.zeros((self.max_num_q))
            len_ques = min(len(ques), self.max_num_q)
            ques_out[:len_ques] = ques[:len_ques]
            ques_out = torch.from_numpy(ques_out).long()
            ques_anno = torch.zeros(self.max_num_q, 1)

            entity_dict = {}
            for q in ques:
                entity_dict[q] = len(entity_dict)
            adj_mat_ques = create_adjacency_matrix_qgraph(
                self.max_num_q, self.n_edge, entity_dict, ques
            )

            for qid in self.data["qid"][idx]:
                if qid in ques:
                    ques_anno[entity_dict[qid]] = 1

            kgs = self.data["kg"][idx]
            kgs_flat = [entity for kg in kgs for entity in kg if entity >= 20]
            total_entity_list = list(set(kgs_flat))

            entity_dict = {}
            for s in total_entity_list:
                entity_dict[s] = len(entity_dict)

            adj_mat_kg = create_adjacency_matrix_kgraph(
                self.max_n_node, self.n_edge, entity_dict, kgs
            )

            kg_anno = torch.zeros(self.max_n_node, 1).float()
            for qid in self.data["qid"][idx]:
                kg_anno[entity_dict[qid]] = 1

            kg_out = np.zeros((self.max_n_node))
            len_entity = min(len(total_entity_list), self.max_n_node)
            kg_out[:len_entity] = total_entity_list[:len_entity]
            kg_out = torch.from_numpy(kg_out).long()

            ans = self.data["ans"][idx]
            return ques_out, adj_mat_ques, ques_anno, kg_out, adj_mat_kg, kg_anno, ans

        elif self.args.model_name == "han":
            ques = self.data["ques"][idx]
            adj_mat_ques = np.zeros((len(ques), len(ques)))

            qadj_idx_row = np.arange(0, len(ques) - 1)
            qadj_idx_col = np.arange(1, len(ques))
            adj_mat_ques[qadj_idx_row, qadj_idx_col] = 1
            adj_mat_ques[qadj_idx_col, qadj_idx_row] = 1

            he_ques_idxs = he_sampling(
                adj_mat_ques, self.max_num_hqnode, self.max_num_q
            )

            kgs = self.data["kg"][idx]
            kgs_flat = [entity for kg in kgs for entity in kg]
            total_entity_list = list(set(kgs_flat))

            entity_dict = {}
            for s in total_entity_list:
                entity_dict[s] = len(entity_dict)

            n_node = len(entity_dict)
            adj_mat_kg = create_adjacency_matrix_flattengraph(n_node, entity_dict, kgs)

            he_kg_idxs = he_sampling(adj_mat_kg, self.max_num_hknode, self.max_num_hk)

            ques = np.array(ques)
            ques_out = np.zeros((self.max_num_q, self.max_num_hqnode))
            for i, he in enumerate(he_ques_idxs):
                ques_out[i, : len(he)] = ques[he[: len(he)]]
            ques_out = torch.from_numpy(ques_out).long()

            total_entity_list = np.array(total_entity_list)
            kg_out = np.zeros((self.max_num_hk, self.max_num_hknode))
            for i, he in enumerate(he_kg_idxs):
                kg_out[i, : len(he)] = total_entity_list[he[: len(he)]]
            kg_out = torch.from_numpy(kg_out).long()

            ans = self.data["ans"][idx]
            return ques_out, kg_out, ans

        elif self.args.model_name == "gcn":
            ques = self.data["ques"][idx]
            ques_out = np.zeros((self.max_num_q))
            len_ques = min(len(ques), self.max_num_q)
            ques_out[:len_ques] = ques[:len_ques]
            ques_out = torch.from_numpy(ques_out).long()

            adj_mat_ques = np.zeros((self.max_num_q, self.max_num_q))
            qadj_idx_row = np.arange(0, len(ques) - 1)
            qadj_idx_col = np.arange(1, len(ques))
            adj_mat_ques[qadj_idx_row, qadj_idx_col] = 1
            adj_mat_ques[qadj_idx_col, qadj_idx_row] = 1

            kgs = self.data["kg"][idx]
            kgs_flat = [entity for kg in kgs for entity in kg]
            total_entity_list = list(set(kgs_flat))

            entity_dict = {}
            for s in total_entity_list:
                entity_dict[s] = len(entity_dict)

            n_node = len(entity_dict)
            adj_mat_kg = create_adjacency_matrix_flattengraph(
                n_node, entity_dict, kgs, self.max_n_node
            )

            kg_out = np.zeros((self.max_n_node))
            len_entity = min(len(total_entity_list), self.max_n_node)
            kg_out[:len_entity] = total_entity_list[:len_entity]
            kg_out = torch.from_numpy(kg_out).long()

            ans = self.data["ans"][idx]
            return ques_out, adj_mat_ques, kg_out, adj_mat_kg, ans

        elif self.args.model_name == "ht" or self.args.model_name == "memnet":
            ques = self.data["ques"][idx]
            ques_he = np.array(list(ngrams(ques, self.max_num_hqnode)))
            ques_out = np.zeros((self.max_num_q, self.max_num_hqnode))
            ques_out[: ques_he.shape[0]] = ques_he
            ques_out = torch.from_numpy(ques_out).long()

            kgs = self.data["kg"][idx]
            if len(kgs) > self.max_num_hk:
                random.shuffle(kgs)
            numiter_kg = min(len(kgs), self.max_num_hk)
            kg_out = np.zeros((self.max_num_hk, self.max_num_hknode))

            for i in range(numiter_kg):
                kg = np.array(kgs[i])
                kg_out[i, : kg.shape[0]] = kg
            kg_out = torch.from_numpy(kg_out).long()
            ans = self.data["ans"][idx]

            return ques_out, kg_out, ans

        elif self.args.model_name == "ban" or self.args.model_name == "ht_abl_wohe":
            ques = self.data["ques"][idx]
            ques_out = np.zeros((self.max_num_q))
            len_ques = min(len(ques), self.max_num_q)
            ques_out[:len_ques] = ques[:len_ques]
            ques_out = torch.from_numpy(ques_out).long()

            kgs = self.data["kg"][idx]
            kgs = list(itertools.chain(*kgs))

            if len(kgs) > self.max_num_hk:
                random.shuffle(kgs)
            numiter_kg = min(len(kgs), self.max_num_hk)
            kg_out = np.zeros((self.max_num_hk))

            for i in range(numiter_kg):
                kg_out[i] = kgs[i]
            kg_out = torch.from_numpy(kg_out).long()
            ans = self.data["ans"][idx]

            return ques_out, kg_out, ans

        elif self.args.model_name == "ht_abl_qset_khe":
            ques = self.data["ques"][idx]
            ques_out = np.zeros((self.max_num_q))
            len_ques = min(len(ques), self.max_num_q)
            ques_out[:len_ques] = ques[:len_ques]
            ques_out = torch.from_numpy(ques_out).long()

            kgs = self.data["kg"][idx]
            if len(kgs) > self.max_num_hk:
                random.shuffle(kgs)
            numiter_kg = min(len(kgs), self.max_num_hk)
            kg_out = np.zeros((self.max_num_hk, self.max_num_hknode))

            for i in range(numiter_kg):
                kg = np.array(kgs[i])
                kg_out[i, : kg.shape[0]] = kg
            kg_out = torch.from_numpy(kg_out).long()
            ans = self.data["ans"][idx]

            return ques_out, kg_out, ans

        elif self.args.model_name == "ht_abl_qhe_kset":
            ques = self.data["ques"][idx]
            ques_he = np.array(list(ngrams(ques, self.max_num_hqnode)))
            ques_out = np.zeros((self.max_num_q, self.max_num_hqnode))
            ques_out[: ques_he.shape[0]] = ques_he
            ques_out = torch.from_numpy(ques_out).long()

            kgs = self.data["kg"][idx]
            kgs = list(itertools.chain(*kgs))

            if len(kgs) > self.max_num_hk:
                random.shuffle(kgs)
            numiter_kg = min(len(kgs), self.max_num_hk)
            kg_out = np.zeros((self.max_num_hk))

            for i in range(numiter_kg):
                kg_out[i] = kgs[i]
            kg_out = torch.from_numpy(kg_out).long()
            ans = self.data["ans"][idx]

            return ques_out, kg_out, ans

    def prepare_instance(self, datum, key, idx):
        self.img_idx.append(key)

        ques = self.proc_data[self.ques_type][key][idx]
        self.ques.append(ques)
        self.wcap.append(self.proc_data["wcap"][key])

        if "det" in self.args.cfg and self.args.model_name == "ggnn":
            self.qid.append(self.detected_qid[key])
        else:
            self.qid.append(self.proc_data["qid"][key])

        self.qtype.append(self.proc_data["qtype"][key][idx])
        kg_all = self.kghop_data[key] + self.kgspat_data[key]
        self.kg.append(kg_all)

        if self.args.model_name == "han" or self.args.model_name == "gcn":
            kgs_flat = [entity for kg in kg_all for entity in kg]
        else:
            kgs_flat = [entity for kg in kg_all for entity in kg if entity >= 20]
        total_entity_list = list(set(ques).union(set(kgs_flat)))
        self.n_node.append(len(total_entity_list))

        answer = str(datum["Answers"][idx])
        ans_idx = self.preprocess_answer(answer)

        if ans_idx == -1:
            answer = answer.encode("raw_unicode_escape").decode("utf-8")
        ans_idx = self.preprocess_answer(answer)

        if self.args.selected != True:
            assert ans_idx != -1
        self.ans.append(np.array(ans_idx))

    def preprocess_answer(self, answer):
        if answer not in self.qid_list:
            answer = answer.lower()

        if answer in self.ne2qid:
            answer = self.ne2qid[answer]

        if answer in self.av2i:
            ans_idx = self.av2i[answer]

        else:
            ans_idx = -1

        return ans_idx

    def create_ans_mask(self):
        one_shot_ans_words = load_files("data/kvqa/processed/one_shot_ans_test.pkl")
        zero_shot_ans_words = load_files("data/kvqa/processed/zero_shot_ans_test.pkl")
        oshot_ans_idxs = []
        zshot_ans_idxs = []
        for word in one_shot_ans_words:
            oshot_ans_idxs.append(self.preprocess_answer(word))

        for word in zero_shot_ans_words:
            zshot_ans_idxs.append(self.preprocess_answer(word))

        return oshot_ans_idxs, zshot_ans_idxs

    def load_data(self, mode, task_idx):
        tic = time.time()
        data = load_files(self.cfg["DATASET"]["RAW_DATA"])
        self.proc_data = load_files(self.cfg["DATASET"]["PROC_DATA"])
        self.ne2qid = load_files(self.cfg["DATASET"]["NE2QID"])
        qid2ne = load_files(self.cfg["DATASET"]["QID2NE"])
        self.qid_list = list(qid2ne.keys())

        n_hop_cfg = "KG_%shop" % (self.n_hop)
        self.kghop_data = load_files(self.cfg["DATASET"][n_hop_cfg])
        self.kgspat_data = load_files(self.cfg["DATASET"]["KG_spat"])
        self.ques_type = self.args.q_opt + "_ques"

        if "det" in self.args.cfg and self.args.model_name == "ggnn":
            self.detected_qid = load_files(self.cfg["DATASET"]["DET_FID"])

        dataset = {}
        self.ques = []
        self.wcap = []
        self.qid = []
        self.qtype = []
        self.kg = []
        self.n_node = []
        self.ans = []

        ans_dict = {}
        ans_imgidx_dict = {}
        for i, key in enumerate(tqdm(data)):
            datum = data[key]
            n_ques = len(datum["Questions"])
            n_split = len(datum["split"])

            if n_ques >= n_split:
                n_data = n_split
            else:
                n_data = n_ques

            for j in range(n_data):
                if datum["split"][j] == 1 and (mode == "train" or mode == "trainval"):
                    if task_idx < 0:  # for all qtype
                        self.prepare_instance(datum, key, j)
                    else:
                        if (
                            task_idx in self.proc_data["qtype"][key][j]
                        ):  # for specific qtype (0, ..., 9)
                            self.prepare_instance(datum, key, j)
                elif datum["split"][j] == 2 and (
                    mode == "val" or mode == "trainval" or mode == "valtest"
                ):
                    if task_idx < 0:  # for all qtype
                        self.prepare_instance(datum, key, j)
                    else:
                        if (
                            task_idx in self.proc_data["qtype"][key][j]
                        ):  # for specific qtype (0, ..., 9)
                            self.prepare_instance(datum, key, j)
                elif datum["split"][j] == 3 and (mode == "test" or mode == "valtest"):
                    if task_idx < 0:  # for all qtype
                        self.prepare_instance(datum, key, j)
                    else:
                        if (
                            task_idx in self.proc_data["qtype"][key][j]
                        ):  # for specific qtype (0, ..., 9)
                            self.prepare_instance(datum, key, j)

        dataset["ques"] = self.ques
        dataset["wcap"] = self.wcap
        dataset["qid"] = self.qid
        dataset["kg"] = self.kg
        dataset["ans"] = self.ans

        print("loaded dataset {}s".format(time.time() - tic))
        return dataset
