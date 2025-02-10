import os
import random
import re
import time
from copy import deepcopy

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import preprocess
import utils


class FlairDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.X.shape[0]


class Data:
    
    def __init__(self, args, verbose=False):
        create_files_df_if_not_exists(args.data_dir, args.files_df_filename)
        self.all_files_df = read_files_df(args.data_dir, args.files_df_filename)
        create_files_numpy_if_not_exists(args.data_dir,
                                         args.files_np_filename,
                                         self.all_files_df)
        self.all_files_np = read_files_np(args.data_dir, args.files_np_filename)
        self.normalizer = preprocess.Normalizer()
        self.saved_datasets = dict()
        self.verbose = verbose
        self.parts = []
    
    def numpy_datasets(self, args):
        train_parts = [i for i in range(1, 6) if i not in [args.test_part]]
        
        X_train_np, y_train_np = self.numpy_dataset(train_parts,
                                                    args.normalization_mode,
                                                    args.data_dir,
                                                    args.train_k,
                                                    args.train_n,
                                                    train=True,
                                                    nan_mode=args.nan_mode,
                                                    binary=args.binary)
        if self.verbose:
            self.print_stats("train", y_train_np, args.binary)
        
        if args.val_p is not None:
            val_idx = np.random.choice(int(len(X_train_np)),
                                       int(len(X_train_np) * args.val_p),
                                       replace=False)
            val_mask = np.zeros(len(X_train_np), dtype=bool)
            val_mask[val_idx] = True
            X_val_np = deepcopy(X_train_np[val_mask])
            y_val_np = deepcopy(y_train_np[val_mask])
            X_train_np = X_train_np[~val_mask]
            y_train_np = y_train_np[~val_mask]
        else:
            X_val_np, y_val_np = None, None
        
        X_test_np, y_test_np = self.numpy_dataset([args.test_part],
                                                  args.normalization_mode,
                                                  args.data_dir,
                                                  nan_mode=args.nan_mode,
                                                  binary=args.binary,
                                                  cache=args.cache)
        if self.verbose:
            self.print_stats("test ", y_test_np, args.binary)
        
        return X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np
    
    def numpy_dataset(self, parts, normalization_mode, data_dir, k=None, n=None,
                      train=False, nan_mode=None, binary=True, cache=False):
        hash_dataset = f"{utils.hash_dataset(parts, k, n, nan_mode, binary)}"
        dataset_loc = os.path.join(data_dir, hash_dataset)
        if cache and hash_dataset in self.saved_datasets:
            (X, y, norm_vals) = deepcopy(self.saved_datasets[hash_dataset])
            if train:
                self.normalizer = preprocess.Normalizer(vals=norm_vals)
        # elif cache and n is None and k is None and os.path.exists(f"{dataset_loc}.npz"):
        #     with np.load(f"{dataset_loc}.npz") as np_data:
        #         X = np_data['X']
        #         y = np_data['y']
        #         norm_vals = np_data['norm_vals']
        #     X, y = self.preprocess(X, y, normalization_mode, train)
        #     self.saved_datasets[hash_dataset] = (X, y, norm_vals)
        else:
            files_df = split(self.all_files_df, partitions=parts, k=k, n=n,
                             binary=binary)
            X, y = read_instances(files_df, self.all_files_np, binary)
            X, y = preprocess.nan_to_num(X, y, nan_mode)
            self.normalizer.fit(X)
            norm_vals = self.normalizer.values()
            # if not cache and n is None and k is None:
            #     np.savez(dataset_loc, X=X, y=y, norm_vals=norm_vals)
            X, y = self.preprocess(X, y, normalization_mode, train)
            if cache:
                self.saved_datasets[hash_dataset] = (X, y, norm_vals)
        return X, y
    
    def preprocess(self, X, y, normalization_mode, train=False):
        start_time = time.time()
        if train:
            self.normalizer = preprocess.Normalizer()
            X_norm = self.normalizer.fit_transform(X, normalization_mode)
        else:
            X_norm = self.normalizer.transform(X, normalization_mode)
        p = np.random.permutation(len(X_norm))
        X_norm, y = X_norm[p], y[p]
        return X_norm, y
    
    @staticmethod
    def print_stats(prefix, y, binary):
        stats = statistics(y, n_class=2 if binary else 4)
        a, portion = stats["n"], stats["portion"]
        follow_up = ["BCQ", "MX"] if binary else ["Q", "BC", "M", "X"]
        print(f"{prefix}: {a:5d} all, {list(zip(portion, follow_up))}")
    
    @staticmethod
    def dataholders(args, X_train_np, y_train_np, X_val_np, y_val_np,
                    X_test_np, y_test_np, test=False):
        X_train = torch.tensor(X_train_np, dtype=torch.float32, device=args.device)
        y_train = torch.tensor(y_train_np, dtype=torch.long, device=args.device)
        if args.batch_size is None or args.batch_size == 0.0:
            train = [utils.DataPair(X_train, y_train)]
        else:
            train_dataset = FlairDataset(X=X_train, y=y_train)
            train = DataLoader(train_dataset,
                               batch_size=args.batch_size,
                               shuffle=True)
        if test:
            val = None
        else:
            X_val = torch.tensor(X_val_np, dtype=torch.float32, device=args.device)
            y_val = torch.tensor(y_val_np, dtype=torch.long, device=args.device)
            val = [utils.DataPair(X_val, y_val)]
        test_set_size = args.batch_size if args.batch_size is not None else 1024
        X_test = torch.split(torch.tensor(X_test_np, dtype=torch.float32), test_set_size)
        y_test = torch.split(torch.tensor(y_test_np, dtype=torch.long), test_set_size)

        test = [utils.DataPair(X_test[i], y_test[i]) for i in range(len(X_test))]
        return train, val, test


def create_partition_files_df(partition_dir, files_df_path=None):
    data = {"path": [],
            "active_region": [],
            "partition": [],
            "label": []}
    
    for flare in os.listdir(partition_dir):
        flare_path = os.path.join(partition_dir, flare)
        for file_name in os.listdir(flare_path):
            path = os.path.join(
                os.path.join(*(flare_path.split(os.path.sep)[1:])), file_name)
            label = file_name[0] if file_name[0] != "F" else "Q"
            partition = re.search(r'partition\d+', partition_dir).group()
            active_region = re.search(r"ar\d+", file_name).group()
            data["path"].append(path)
            data["partition"].append(partition)
            data["active_region"].append(active_region)
            data["label"].append(label)
    partition_files_df = pandas.DataFrame(data)
    if files_df_path is not None:
        partition_files_df.to_csv(files_df_path, index=False)
    return partition_files_df


def create_files_df(partition_dirs, file_path):
    files_df = pandas.DataFrame()
    for partition_dir in partition_dirs:
        partition_files_df = create_partition_files_df(partition_dir)
        files_df = pandas.concat([files_df, partition_files_df])
    files_df.to_csv(file_path, index=False)
    return files_df


def create_files_numpy(data_dir, file_path, files_df):
    n = len(files_df)
    pbar = tqdm(total=n, unit="files", smoothing=0.01)
    X = np.empty((n, 24, 60))
    for index, row in files_df.iterrows():
        file_path = row["path"]
        df = pandas.read_csv(os.path.join(data_dir, file_path), delimiter="\t")
        df = df[df.columns[1:25]]
        X[index] = df.to_numpy().T
        pbar.update(1)
    np.save(file_path, X)


def choose_from_active_regions(files_df, k):
    indices = []
    active_regions = files_df.active_region.unique()
    k = k / len(active_regions)
    for active_region in active_regions:
        active_region_df = files_df[files_df.active_region == active_region]
        n = int(k)
        if random.random() < k - int(k):
            # with remaining probability select one more
            n += 1
        indices.extend(active_region_df.sample(n=min(n, len(active_region_df)),
                                               replace=False).index)
    return files_df.loc[indices]


def split(files_df, partitions, k=None, n=None, binary=True):
    if binary:
        bcq = files_df.query(
            f"(partition == {[f'partition{i}' for i in partitions]})"
            f" and (label == ['B', 'C', 'Q'])")
        mx = files_df.query(
            f"(partition == {[f'partition{i}' for i in partitions]})"
            f" and (label == ['M', 'X'])")
        if n is not None:
            bcq = bcq.sample(n=min(n[0], len(bcq)), replace=False)
            mx = mx.sample(n=min(n[1], len(mx)), replace=False)
        elif k is not None:
            bcq = choose_from_active_regions(bcq, k=k[0])
            mx = choose_from_active_regions(mx, k=k[1])
        return pandas.concat([bcq, mx])
    else:
        q = files_df.query(
            f"(partition == {[f'partition{i}' for i in partitions]})"
            f" and (label == ['Q'])")
        bc = files_df.query(
            f"(partition == {[f'partition{i}' for i in partitions]})"
            f" and (label == ['B', 'C'])")
        m = files_df.query(
            f"(partition == {[f'partition{i}' for i in partitions]})"
            f" and (label == ['M'])")
        x = files_df.query(
            f"(partition == {[f'partition{i}' for i in partitions]})"
            f" and (label == ['X'])")
        if n is not None:
            q = q.sample(n=min(n[0], len(q)), replace=False)
            bc = bc.sample(n=min(n[1], len(bc)), replace=False)
            m = m.sample(n=min(n[2], len(m)), replace=False)
            x = x.sample(n=min(n[3], len(x)), replace=False)
        elif k is not None:
            q = choose_from_active_regions(q, k=k[0])
            bc = choose_from_active_regions(bc, k=k[1])
            m = choose_from_active_regions(m, k=k[2])
            x = choose_from_active_regions(x, k=k[3])
        return pandas.concat([q, bc, m, x])


def statistics(np_array, n_class):
    stats = dict()
    stats["portion"] = []
    for i in range(n_class):
        stats["portion"].append((np_array == i).sum())
    stats["n"] = np_array.shape[0]
    return stats


def read_instances(files_df, files_np, binary):
    i, n = 0, len(files_df)
    X, y = np.empty((n, 24, 60)), np.empty(n)
    
    pbar = tqdm(total=n, unit="files", smoothing=0.01, disable=True)
    for index, row in files_df.iterrows():
        if binary:
            label = 0 if (row["label"] in ["Q", "B", "C"]) else 1
        elif row["label"] == "Q":
            label = 0
        elif row["label"] in ["B", "C"]:
            label = 1
        elif row["label"] in ["M"]:
            label = 2
        else:
            label = 3
        
        y[i] = label
        X[i] = deepcopy(files_np[index])
        i += 1
        pbar.update(1)
    return X, y


def create_files_df_if_not_exists(data_dir, files_df_filename):
    if not os.path.exists(os.path.join(data_dir, files_df_filename)):
        print("Creating all files df ...")
        partition_dirs = []
        for partition_dir in os.listdir(data_dir):
            if "partition" not in partition_dir:
                continue
            partition_dirs.append(os.path.join(data_dir, partition_dir))
            create_files_df(partition_dirs,
                            os.path.join(data_dir, files_df_filename))
        print("Created all files df")


def create_files_numpy_if_not_exists(data_dir, files_np_filename, files_df):
    print(os.path.join(data_dir, files_np_filename))
    if not os.path.exists(os.path.join(data_dir, files_np_filename)):
        print("Creating all files numpy ...")
        create_files_numpy(data_dir,
                           os.path.join(data_dir, files_np_filename),
                           files_df)


def read_files_df(data_dir, files_df_filename):
    print("Reading all files df ...")
    return pandas.read_csv(os.path.join(data_dir, files_df_filename))


def read_files_np(data_dir, files_np_filename):
    print("Reading all files np ...")
    return np.load(os.path.join(data_dir, files_np_filename))
