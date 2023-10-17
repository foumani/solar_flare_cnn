from copy import deepcopy

import preprocess


class Context:
    data_dir = None
    tensorlog_dir = None
    log_dir = None
    model_dir = None
    files_df_filename = None
    files_np_filename = None
    split_report_filename = None
    model_report_filename = None
    device = "cpu"
    
    def __init__(self,
                 binary=True,
                 train_parts=None,
                 val_part=None,
                 test_part=None,
                 train_k=None,
                 train_n=None,
                 ch_conv1=None,
                 ch_conv2=None,
                 ch_conv3=None,
                 l_hidden=None,
                 batch_size=None,
                 nan_mode=None,
                 normalization_mode=preprocess.Normalizer.z_score,
                 data_dropout=0.0,
                 layer_dropout=0.0,
                 class_importance=None,
                 lr=0.01,
                 early_stop=10,
                 stop=None,
                 run_times=1,
                 run_no=0,
                 n_random_search=None,
                 val_p=None,
                 draw=False,
                 ablation=False
                 ):
        self.binary = binary
        self.train_parts = train_parts
        self.val_part = val_part
        self.test_part = test_part
        self.train_k = train_k
        self.train_n = train_n
        self.ch_conv1 = ch_conv1
        self.ch_conv2 = ch_conv2
        self.ch_conv3 = ch_conv3
        self.l_hidden = l_hidden
        self.batch_size = batch_size
        self.nan_mode = nan_mode
        self.normalization_mode = normalization_mode
        self.data_dropout = data_dropout
        self.layer_dropout = layer_dropout
        self.class_importance = class_importance
        self.lr = lr
        self.early_stop = early_stop
        self.stop = stop
        self.run_times = run_times
        self.run_no = run_no
        self.n_random_search = n_random_search
        self.val_p = val_p
        self.draw = draw
        self.ablation = ablation
    
    def __copy__(self):
        return Context(binary=self.binary,
                       train_parts=deepcopy(self.train_parts),
                       val_part=self.val_part,
                       test_part=self.test_part,
                       train_k=self.train_k,
                       train_n=self.train_n,
                       ch_conv1=self.ch_conv1,
                       ch_conv2=self.ch_conv2,
                       ch_conv3=self.ch_conv3,
                       l_hidden=self.l_hidden,
                       batch_size=self.batch_size,
                       nan_mode=self.nan_mode,
                       normalization_mode=self.normalization_mode,
                       data_dropout=self.data_dropout,
                       layer_dropout=self.layer_dropout,
                       class_importance=self.class_importance,
                       lr=self.lr,
                       early_stop=self.early_stop,
                       stop=self.stop,
                       run_times=self.run_times,
                       run_no=self.run_no,
                       n_random_search=self.n_random_search,
                       val_p=self.val_p,
                       draw=self.draw)
    
    def __repr__(self):
        vals = {
            "binary": self.binary,
            "val part": self.val_part if self.val_part is not None else self.val_p,
            "test part": self.test_part,
            "train mode": "n" if self.train_n is not None else "k",
            "split": self.train_n if self.train_n is not None else self.train_k,
            "conv layers": [self.ch_conv1, self.ch_conv2, self.ch_conv3],
            "fcn": [self.l_hidden],
            "batch size": self.batch_size,
            "nan mode": self.nan_mode,
            "normalization mode": self.normalization_mode,
            "dropouts": [self.data_dropout, self.layer_dropout],
            "class importance": self.class_importance,
            "lr": self.lr,
            "# random search": self.n_random_search,
        }
        return (f"{Context.__name__}("
                f"{vals},"
                f")")
