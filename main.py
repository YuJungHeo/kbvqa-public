import os
import time
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    load_files,
    save_pickle,
    fix_seed,
    print_model,
    CosineAnnealingWarmUpRestarts,
)
from model import (
    BAN,
    HAN,
    GGNN,
    GCN,
    MemNet,
    HypergraphTransformer,
    HypergraphTransformer_wohe,
    HypergraphTransformer_qsetkhe,
    HypergraphTransformer_qhekset,
)
from modules.logger import setup_logger, get_rank
from dataloader import KVQA, PQnPQL, load_PQnPQL_data, FVQA, load_FVQA_data


def eval_epoch(model, loader, args):
    model.eval()
    total_right = 0
    total_right_aset = 0
    total_num = 0

    for b_idx, batch in enumerate(tqdm(loader)):
        batch = [b.cuda() for b in batch]
        labels = batch[-1]

        pred = model(batch)
        pred_score, pred_ans = pred.max(1)

        nz_idxs = labels.nonzero()
        right = labels[nz_idxs] == pred_ans[nz_idxs]
        total_right += right.sum().item()
        total_num += len(labels)

        if "fvqa" in args.data_name:
            _, top3_indices = torch.topk(pred, 3)
            for idx, indices in enumerate(top3_indices):
                if labels[idx] in indices:
                    total_right_aset += 1

        if "pq" in args.data_name:
            aset = batch[-2]
            for idx, pred in enumerate(pred_ans):
                if pred in aset[idx]:
                    total_right_aset += 1

    return total_right, total_right_aset, total_num


def inference(model, test_loader, ckpt_path, args, task_idx=-1, res=False):
    last_ckpt = os.path.join(ckpt_path, "ckpt_best.pth.tar")
    checkpoint = torch.load(last_ckpt)

    if list(checkpoint["state_dict"].keys())[0].startswith("module."):
        checkpoint["state_dict"] = {
            k[7:]: v for k, v in checkpoint["state_dict"].items()
        }

    model.load_state_dict(checkpoint["state_dict"])
    print("load: %s" % (last_ckpt))

    total_right, total_right_aset, total_num = eval_epoch(model, test_loader, args)
    accuracy = total_right / total_num

    if "pq" in args.data_name or "fvqa" in args.data_name:
        if "fvqa" in args.data_name:
            print("## Test accuracy (@1) : %f" % (accuracy))
        accuracy = total_right_aset / total_num
        
    return accuracy


def main():
    """parse config file"""
    parser = argparse.ArgumentParser(description="experiments")
    parser.add_argument("--model_name", default="ht")
    parser.add_argument("--data_name", default="kvqa")
    parser.add_argument("--cfg", default="ht")
    parser.add_argument("--exp_name", default="dev")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--per_cate", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--selected", action="store_true")
    parser.add_argument("--abl_only_ga", action="store_true")
    parser.add_argument("--abl_only_sa", action="store_true")
    parser.add_argument("--abl_ans_fc", action="store_true")
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--q_opt", type=str, default="org")
    parser.add_argument("--n_hop", type=int, default=1)
    args = parser.parse_args()

    config_file = "configs/%s.yaml" % (args.cfg)
    model_cfg = load_files(config_file)

    fix_seed(model_cfg["MODEL"]["SEED"])

    if args.debug == False:
        summary_path = model_cfg["RES"]["TB"] + args.exp_name
        summary = SummaryWriter(summary_path)

    log_path = model_cfg["RES"]["LOG"] + args.exp_name
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = model_cfg["RES"]["CKPT"] + args.exp_name
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    logger = setup_logger(args.exp_name, log_path, get_rank())
    logger.info(model_cfg["MODEL"])
    logger.info(args)

    # ------------ Construct Dataset Class ------------------------------------
    datasets = {}
    if args.data_name == "kvqa":
        modes = ["train", "val", "test"]
        n_node_lists = []
        for mode in modes:
            fname = ckpt_path + "/%s_cache.pkl" % (mode)
            if os.path.isfile(fname):
                datasets[mode] = load_files(fname)
            else:
                data = KVQA(model_cfg, args, mode)
                datasets[mode] = data
                save_pickle(data, fname)
            n_node_lists.append(max(datasets[mode].n_node))
        max_n_node = max(n_node_lists)

        for mode in modes:
            datasets[mode].max_n_node = max_n_node

    elif "fvqa" in args.data_name:
        train, test = load_FVQA_data(model_cfg, args)
        datasets["train"] = FVQA(model_cfg, args, train)
        datasets["test"] = FVQA(model_cfg, args, test)

    elif "pq" in args.data_name:
        train, val, test = load_PQnPQL_data(model_cfg, args)
        datasets["train"] = PQnPQL(model_cfg, args, train)
        datasets["val"] = PQnPQL(model_cfg, args, val)
        datasets["test"] = PQnPQL(model_cfg, args, test)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=model_cfg["MODEL"]["BATCH_SIZE"],
        num_workers=args.num_workers,
        shuffle=True,
    )
    if "fvqa" in args.data_name:
        val_loader = DataLoader(
            datasets["test"],
            batch_size=model_cfg["MODEL"]["BATCH_SIZE"],
            num_workers=args.num_workers,
            shuffle=True,
        )
    else:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=model_cfg["MODEL"]["BATCH_SIZE"],
            num_workers=args.num_workers,
            shuffle=True,
        )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=model_cfg["MODEL"]["BATCH_SIZE"],
        num_workers=args.num_workers,
        shuffle=False,
    )

    # ------------ Model -----------------------
    if args.model_name == "ht":
        model = HypergraphTransformer(model_cfg, args).cuda()
    elif args.model_name == "ht_abl_wohe":
        model = HypergraphTransformer_wohe(model_cfg, args).cuda()
    elif args.model_name == "ht_abl_qset_khe":
        model = HypergraphTransformer_qsetkhe(model_cfg, args).cuda()
    elif args.model_name == "ht_abl_qhe_kset":
        model = HypergraphTransformer_qhekset(model_cfg, args).cuda()
    elif args.model_name == "ggnn":
        model = GGNN(model_cfg, args, max_n_node).cuda()
    elif args.model_name == "han":
        model = HAN(model_cfg, args).cuda()
    elif args.model_name == "ban":
        model = BAN(model_cfg, args).cuda()
    elif args.model_name == "memnet":
        model = MemNet(model_cfg, args).cuda()
    elif args.model_name == "gcn":
        model = GCN(model_cfg, args).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=150, T_mult=1, eta_max=0.001, T_up=10, gamma=0.5
    )
    model.cuda()

    # ------------ Evaluate -----------------------
    if args.inference == True:
        if args.per_cate == False:
            test_acc_final = inference(model, test_loader, ckpt_path, args, res=False)
            logger.info("test accuracy (final) : %f" % (test_acc_final))

        else:  # analysis on question types (KVQA only)
            if args.data_name == "kvqa":
                cate_accu_test = []
                qtypes = load_files(model_cfg["DATASET"]["IDX2QTYPE"])
                for task_idx in range(10):
                    test = KVQA(model_cfg, args, "test", task_idx)
                    test.max_n_node = max_n_node
                    test_loader = DataLoader(
                        test,
                        batch_size=model_cfg["MODEL"]["BATCH_SIZE"],
                        num_workers=args.num_workers,
                        shuffle=False,
                    )
                    accu = inference(
                        model, test_loader, ckpt_path, args, task_idx=task_idx, res=True
                    )
                    cate_accu_test.append(accu)
                print(qtypes[:10])
                print(cate_accu_test)
            else:
                raise NotImplementedError(
                    "Datasets except KVQA do not have categories for questions. Set per_cate as False."
                )
        return 0

    # ------------ Training -----------------------
    train_loss = []
    best_acc = 0.0

    for e_idx in range(0, args.max_epoch):
        model.train()
        total_right = 0
        total_num = 0
        total_right_aset = 0
        for b_idx, batch in enumerate(tqdm(train_loader)):
            batch = [b.cuda() for b in batch]
            labels = batch[-1]
            pred = model(batch)
            pred_score, pred_ans = pred.max(1)
            loss = F.nll_loss(pred, labels)
            train_loss.append(loss.item())

            nz_idxs = labels.nonzero()
            right = labels[nz_idxs] == pred_ans[nz_idxs]
            total_right += right.sum().item()
            total_num += len(labels)

            if "fvqa" in args.data_name:
                _, top3_indices = torch.topk(pred, 3)
                for idx, indices in enumerate(top3_indices):
                    if labels[idx] in indices:
                        if labels[idx] != 0:
                            total_right_aset += 1  # top-3 accuracy

            if "pq" in args.data_name:
                aset = batch[-2]
                for idx, pred in enumerate(pred_ans):
                    if pred in aset[idx]:
                        total_right_aset += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.debug == False:
                summary.add_scalar(
                    "loss/train", loss.item(), e_idx * len(train_loader) + b_idx
                )

        if args.schedule:
            lr_scheduler.step()

        if args.debug == False:
            tr_accu = total_right / total_num
            tr_accu_aset = total_right_aset / total_num
            summary.add_scalar("accu/train", tr_accu, e_idx)

            if "pq" in args.data_name:
                summary.add_scalar("accu_aset/train", tr_accu_aset, e_idx)
                logger.info(
                    "epoch %i train accuracy : %f, %i/%i / %f, %i/%i "
                    % (
                        e_idx,
                        tr_accu,
                        total_right,
                        total_num,
                        tr_accu_aset,
                        total_right_aset,
                        total_num,
                    )
                )
            else:
                logger.info(
                    "epoch %i train accuracy : %f, %i/%i"
                    % (e_idx, tr_accu, total_right, total_num)
                )

        with torch.no_grad():
            total_right_val, total_right_aset_val, total_num_val = eval_epoch(
                model, val_loader, args
            )

        if args.debug == False:
            val_acc = total_right_val / total_num_val
            val_acc_aset = total_right_aset_val / total_num_val
            summary.add_scalar("accu/val", val_acc, e_idx)

            if "pq" in args.data_name or "fvqa" in args.data_name:
                summary.add_scalar("accu_aset/val", val_acc_aset, e_idx)
                logger.info(
                    "epoch %i val accuracy : %f, %i/%i / %f, %i/%i"
                    % (
                        e_idx,
                        val_acc,
                        total_right_val,
                        total_num_val,
                        val_acc_aset,
                        total_right_aset_val,
                        total_num_val,
                    )
                )
                if "pq" in args.data_name:
                    val_acc = val_acc_aset
            else:
                logger.info(
                    "epoch %i val accuracy : %f, %i/%i"
                    % (e_idx, val_acc, total_right_val, total_num_val)
                )

            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "epoch_idx": e_idx,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(ckpt_path, "ckpt_best.pth.tar"),
                )
            logger.info("## Current VAL Best : %f" % (best_acc))

    test_acc_final = inference(model, test_loader, ckpt_path, args)
    logger.info("## Test accuracy : %f" % (test_acc_final))
    if "pq" in args.data_name:
        summary.add_scalar("accu_aset/test", test_acc_final, 0)
    elif "fvqa" in args.data_name:
        summary.add_scalar("accu_aset/test", test_acc_final, 0)
        summary.add_scalar("accu/test", test_acc_final, 0)
    else:
        summary.add_scalar("accu/test", test_acc_final, 0)

if __name__ == "__main__":
    main()
