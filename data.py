import os
import random
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from copy import deepcopy

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import preprocess
import utils
from imblearn.over_sampling import SMOTE
from tsaug import AddNoise, Dropout, Quantize
import pandas as pd


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
        self.np_rng = np.random.default_rng(seed=args.seed)
        self.pd_rng_state = np.random.default_rng(seed=args.seed+1)
        self.saved = dict()
        self.parts = []
        self.remove_mostly_bad()
        self.pre_processed = dict()

    def remove_mostly_bad(self):
        """
        Removes the instances where more than 25% of the data is missing
        :return:
        """
        mask = (self.all_files_np[:, :23, :] == 0)
        self.all_files_np[:, :23, :][mask] = np.nan
        print(self.all_files_np.shape)
        print(self.all_files_np[self.all_files_np == 0].shape)
        is_nan_or_zero = (self.all_files_np == 0) | np.isnan(self.all_files_np)
        count_zero_or_nan = is_nan_or_zero.sum(axis=(1, 2))
        is_mostly_good = count_zero_or_nan < (60 * 24 / 8)
        self.all_files_np = self.all_files_np[is_mostly_good]
        self.all_files_df = self.all_files_df[is_mostly_good]
        self.all_files_df = self.all_files_df.reset_index(drop=True)

    def plot_ar_hist(self, args):
        print(self.all_files_df.columns)
        print(self.all_files_df["path"].loc[0])
        print(self.all_files_df["active_region"].head(10))

        counts = self.all_files_df[
            'active_region'].value_counts()  # you can adjust bins as needed
        print(counts)
        print(type(counts))

        # print(counts.hist())
        # print(counts.columns)
        # plt.hist(counts, bins=1)
        counts.hist(bins=24, weights=(np.ones(len(counts)) * 100.0) / len(counts))
        plt.xlabel('Number of instances per active region')
        plt.ylabel('Percentage of active regions')
        plt.title('Distribution of Active Region Instance Frequencies')
        # plt.gca().yaxis.set_major_formatter(PercentFormatter())
        ax = plt.gca()
        ax.yaxis.set_major_formatter(PercentFormatter())
        yticks = ax.get_yticks()
        yticks = [tick for tick in yticks if tick < 11.0]
        ax.set_yticks(yticks)
        plt.savefig("dist_ar_freq_noline.eps", format='eps', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_instance_removal(self, args):
        print(self.all_files_df.columns)
        print(self.all_files_df.head(5))
        print(self.all_files_df["partition"].head(5))
        counts = self.all_files_df["label"].value_counts()
        print(counts)
        print(counts["Q"])

        before = dict()
        after = dict()
        for c in ['partition1', 'partition2', 'partition3', 'partition4', 'partition5']:
            before[c] = {"Q": 0,
                         "B": 0,
                         "C": 0,
                         "M": 0,
                         "X": 0}

            after[c] = {"Q": 0,
                        "B": 0,
                        "C": 0,
                        "M": 0,
                        "X": 0}

        vals = {"Q": 0,
                "B": 0,
                "C": 0,
                "M": 0,
                "X": 0}

        indices = np.isnan(self.all_files_np).any(axis=(1, 2))
        for index, row in self.all_files_df.iterrows():
            y = row["label"]
            partition = row["partition"]
            before[partition][y] += 1
            if indices[index]:
                continue
            after[partition][y] += 1
            if y == "Q":
                vals["Q"] += 1
            elif y == "B":
                vals["B"] += 1
            elif y == "C":
                vals["C"] += 1
            elif y == "M":
                vals["M"] += 1
            elif y == "X":
                vals["X"] += 1

        print(vals)

        labels = ['Q', 'B', 'C', 'M', 'X']

        for partition in ['partition1', 'partition2', 'partition3', 'partition4',
                          'partition5']:
            print("partition", partition)
            print("before", before[partition])
            print("after", after[partition])
            print()

        values1 = [counts[labels[i]] for i in range(5)]
        values2 = [vals[labels[i]] for i in range(5)]

        x = np.arange(len(labels))  # positions of groups on x-axis
        width = 0.35  # width of each bar

        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width / 2, values1, width, label='Before Instance Removal')
        bars2 = ax.bar(x + width / 2, values2, width, label='After Instance Removal')

        # Labels and formatting
        ax.set_xlabel('Flare Class')
        ax.set_ylabel('Count')
        ax.set_title('Number of Flare Classes Before and After Instance Removal')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.tight_layout()
        plt.savefig("instance_removal.eps", format='eps', dpi=300, bbox_inches='tight')
        plt.show()

    def numpy_datasets(self, args, run):  # todo: add saving later on
        self.np_rng = np.random.default_rng(seed=args.seed + run)
        self.pd_rng_state = np.random.default_rng(seed=args.seed + run + 1)

        # read data
        train_parts = [i for i in range(1, 6) if i not in [args.test_part]]
        X_train_np, y_train_np, files_df_train = self.load(args, train_parts)
        # if args.val_p is not None:  # we are always going to use this, there is valp
        val_idx = self.np_rng.choice(int(len(X_train_np)),
                                     int(len(X_train_np) * args.val_p),
                                     replace=False)
        val_mask = np.zeros(len(X_train_np), dtype=bool)
        val_mask[val_idx] = True
        X_val_np = deepcopy(X_train_np[val_mask])
        y_val_np = deepcopy(y_train_np[val_mask])
        X_train_np = X_train_np[~val_mask]
        y_train_np = y_train_np[~val_mask]
        files_df_val = files_df_train[val_mask]
        files_df_train = files_df_train[~val_mask]
        # else:
        #     X_val_np, y_val_np = None, None

        X_train_np = deepcopy(X_train_np)
        y_train_np = deepcopy(y_train_np)
        files_df_train = deepcopy(files_df_train)
        X_val_np = deepcopy(X_val_np)
        y_val_np = deepcopy(y_val_np)
        files_df_val = deepcopy(files_df_val)


        # nan to num
        X_train_np, y_train_np, files_df_train = preprocess.nan_to_num(X_train_np, y_train_np, files_df_train,
                                                       args.nan_mode)
        X_val_np, y_val_np, files_df_val = preprocess.nan_to_num(X_val_np, y_val_np, files_df_val, args.nan_mode)


        test_cache_loc = f"part{args.test_part}_nan{args.nan_mode}"
        if test_cache_loc not in self.pre_processed:
            print(args.nan_mode)
            X_test_np, y_test_np, files_df_test = self.load(args, [args.test_part], full=True)
            X_test_np = deepcopy(X_test_np)
            y_test_np = deepcopy(y_test_np)
            X_test_np, y_test_np, files_df_test = preprocess.nan_to_num(X_test_np, y_test_np, files_df_test, args.nan_mode)
            self.pre_processed[test_cache_loc] = (X_test_np, y_test_np)
        else:
            (X_test_np, y_test_np) = self.pre_processed[test_cache_loc]

        # normalize
        normalizer = preprocess.Normalizer()
        X_train_np = normalizer.fit_transform(X_train_np, args.normalization_mode)
        X_val_np = normalizer.transform(X_val_np, args.normalization_mode)
        X_test_np = normalizer.transform(X_test_np, args.normalization_mode)



        # Near Decision Boundary Sample Removal
        if args.ndbsr:
            print(
                f"There are ({(y_train_np == 0).sum()}, {(y_train_np == 1).sum()}) instances.")
            i, j, n = 0, 0, len(files_df_train)
            X, y = np.empty((n, 24, 60)), np.empty(n)
            mask = np.full(n, False)

            for index, row in files_df_train.iterrows():
                if row["label"] in ["B", "C"]:
                    j += 1
                    continue
                y[i] = y_train_np[j]
                X[i] = X_train_np[j]
                mask[j] = True
                i += 1
                j += 1
            X_train_np = X[0:i]
            y_train_np = y[0:i]
            files_df_train = files_df_train[mask]
            print(
                f"There are ({(y_train_np == 0).sum()}, {(y_train_np == 1).sum()}) instances, {n - y_train_np.shape[0]} removed")


        if args.smote:
            print("in smote")
            smote = SMOTE(random_state=42)
            X_train_flat_np = X_train_np.reshape(X_train_np.shape[0], -1)
            X_train_smote_flat, y_train_smote = smote.fit_resample(X_train_flat_np, y_train_np)
            n_smote = y_train_smote.shape[0]
            X_train_np, y_train_np = X_train_smote_flat.reshape(n_smote, 24, 60), y_train_smote


        if args.aug:
            augmenter = (AddNoise(scale=(0.01, 0.05), seed=args.seed)
                         + Quantize(n_levels=256))
            # %%
            X_train_aug = augmenter.augment(np.transpose(X_train_np, (0, 2, 1)))
            X_train_np = np.transpose(X_train_aug, (0, 2, 1))
        return X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np, files_df_train, files_df_val, None

    def numpy_dataset(self, parts, normalization_mode, args, k=None, n=None,
                      train=False, nan_mode=None, binary=True, cache=False):
        hash_dataset = f"{utils.hash_dataset(parts, k, n, nan_mode, binary)}"
        dataset_loc = os.path.join(args.data_dir, hash_dataset)
        if cache and hash_dataset in self.saved_datasets:
            (X, y, norm_vals) = self.saved_datasets[hash_dataset]
            if train:
                self.normalizer = preprocess.Normalizer(vals=norm_vals)
        elif cache and n is None and k is None and os.path.exists(
                f"{dataset_loc}.npz"):  # this is for test set (necessary?) (definitely not necessary)
            with np.load(f"{dataset_loc}.npz") as np_data:
                X = np_data['X']
                y = np_data['y']
                norm_vals = np_data['norm_vals']
            X, y = self.preprocess(X, y, normalization_mode, train)
            self.saved_datasets[hash_dataset] = (X, y, norm_vals)
        else:
            files_df = split(self.all_files_df, partitions=parts, args=args,
                             rng_state=self.pd_rng_state, k=k, n=n, binary=binary)
            X, y = read_instances(files_df, self.all_files_np, binary)
            X, y = preprocess.nan_to_num(X, y, nan_mode)
            self.normalizer.fit(X)
            norm_vals = self.normalizer.values()
            if not cache and n is None and k is None:
                np.savez(dataset_loc, X=X, y=y, norm_vals=norm_vals)
            X, y = self.preprocess(X, y, normalization_mode, train)
            if cache:
                self.saved_datasets[hash_dataset] = (X, y, norm_vals)
        return X, y

    def load(self, args, parts, full=False):
        hash_dataset = f"{utils.hash_dataset(parts, args.train_n, args.nan_mode, args.binary)}"
        if args.cache and hash_dataset in self.saved_datasets:
            (X, y, files_df) = self.saved_datasets[hash_dataset]
        else:
            files_df = split(self.all_files_df, partitions=parts, args=args,
                             rng_state=self.pd_rng_state,
                             n=args.train_n if not full else None,
                             binary=args.binary)
            X, y = read_instances(files_df, self.all_files_np, args.binary)
            if args.cache:
                self.saved_datasets[hash_dataset] = (X, y, files_df)
        return X, y, files_df

    def preprocess(self, X, y, normalization_mode, train=False):
        if train:
            self.normalizer = preprocess.Normalizer()
            X_norm = self.normalizer.fit_transform(X, normalization_mode)
        else:
            X_norm = self.normalizer.transform(X, normalization_mode)
        p = self.np_rng.permutation(len(X_norm))
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
                    X_test_np, y_test_np, train_df, val_df, test_df, test=False):
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

    @staticmethod
    def select_indices(train, val, test, args):
        selected_indices = args.ordering[:args.n_features]
        train.dataset.X = train.dataset.X[:, selected_indices, :]

        val[0] = utils.DataPair(val[0].X[:, selected_indices, :], val[0].y)

        for i in range(len(test)):
            test[i] = utils.DataPair(test[i].X[:, selected_indices, :], test[i].y)

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


def data_columns(args):
    dir = os.path.join(args.data_dir, "partition1")  # partition
    dir = os.path.join(dir, os.listdir(dir)[0])  # flare
    file = os.path.join(dir, os.listdir(dir)[0])

    df = pandas.read_csv(file, delimiter="\t")
    return df.columns


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

def nonuniform_sample(files_df, k):
    indices = []
    active_regions = files_df.active_region.unique()
    I = dict()
    I_hat = dict()
    for active_region in active_regions:
        I[active_region] = files_df[files_df.active_region == active_region]
        I_hat[active_region] = pd.DataFrame({}, columns=files_df.columns)

    for i in range(k):
        U = np.zeros(len(active_regions))
        for active_region in active_regions:
            U[active_region] = len(I[active_region]) / (len(I_hat[active_region]) + 1)
        U = U / np.sum(U)
        ar = np.random.choice(active_regions, 1, p=U)
        entry = I[ar].sample(n=1, replace=False)
        indices.extend(entry.index)
        entry = I[ar].loc[I[ar]]
        I_hat[ar] = pd.concat([I_hat[ar], entry])
        I[ar] = I[ar].drop([entry.index])

    return files_df.loc[indices]


def split(files_df, partitions, args, rng_state, k=None, n=None, binary=True):
    if binary:
        bcq = files_df.query(f"(partition == {[f'partition{i}' for i in partitions]})"
                             f" and (label == ['B', 'C', 'Q'])")
        mx = files_df.query(f"(partition == {[f'partition{i}' for i in partitions]})"
                            f" and (label == ['M', 'X'])")
        if n is not None:
            bcq = bcq.sample(n=min(n[0], len(bcq)), replace=False, random_state=rng_state)
            mx = mx.sample(n=min(n[1], len(mx)), replace=False, random_state=rng_state)
        elif k is not None:
            bcq = choose_from_active_regions(bcq, k=k[0])
            mx = choose_from_active_regions(mx, k=k[1])
        return pandas.concat([bcq, mx])
    else:
        q = files_df.query(f"(partition == {[f'partition{i}' for i in partitions]})"
                           f" and (label == ['Q'])")
        bc = files_df.query(f"(partition == {[f'partition{i}' for i in partitions]})"
                            f" and (label == ['B', 'C'])")
        m = files_df.query(f"(partition == {[f'partition{i}' for i in partitions]})"
                           f" and (label == ['M'])")
        x = files_df.query(f"(partition == {[f'partition{i}' for i in partitions]})"
                           f" and (label == ['X'])")
        if n is not None:
            q = q.sample(n=min(n[0], len(q)), replace=False, random_state=rng_state)
            bc = bc.sample(n=min(n[1], len(bc)), replace=False, random_state=rng_state)
            m = m.sample(n=min(n[2], len(m)), replace=False, random_state=rng_state)
            x = x.sample(n=min(n[3], len(x)), replace=False, random_state=rng_state)
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
            create_files_df(partition_dirs, os.path.join(data_dir, files_df_filename))
        print("Created all files df")


def create_files_numpy_if_not_exists(data_dir, files_np_filename, files_df):
    print(os.path.join(data_dir, files_np_filename))
    if not os.path.exists(os.path.join(data_dir, files_np_filename)):
        print("Creating all files numpy ...")
        create_files_numpy(data_dir, os.path.join(data_dir, files_np_filename), files_df)


def read_files_df(data_dir, files_df_filename):
    print("Reading all files df ...")
    return pandas.read_csv(os.path.join(data_dir, files_df_filename))


def read_files_np(data_dir, files_np_filename):
    print("Reading all files np ...")
    return np.load(os.path.join(data_dir, files_np_filename))
