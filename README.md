This repository contains the source code for paper `Convolutional Neural
Network-based Solar Flare Prediction using Multivariate Time-Series of the Solar
Magnetic Field Parameters`.

## Installing Packages

Using python version 3.8.10 run the following command.

```bash
pip install -r requirements.txt
```

## Downloading the Dataset

Download the 5 partitions from dataset
from [This link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM).
Extract the partitions and structure data in the following format:

```text
<data_dir>
|--- partition1
|--- partition2
|--- partition3
|--- partition4
|--- partition5
```

## Replicating the Paper's Results

To replicate the paper's results run the following command (Replace `<data_dir>`
with the directory path where you extracted the data).

```bash
python experiments.py --datadir <data_dir> --logdir log
```

The resulting plots are saved in the `plots/` folder.

## Running the Code

To find the best combination of hyper-parameters for the model run the following
commands.

* For binary-class classification:

```bash
python train.py --datadir <data_dir> --logdir log --paramsearch 500
```

* For multi-class classification:

```bash
python train.py --datadir <data_dir> --logdir log --paramsearch 500 --multi
```

The resutls (comparison between different hyper-parameters) are saved
in `plot/model_report-[binary or multi].csv`

To find the best combination of hyper-parameters for the baselines for binary
classification run the following command (Replace <method_name> with svm,
minirocket, lstm, cif, cnn according to which baseline you want to run)

```bash
python baselines.py --method <method_name> --datadir <datadir> --logdir log --paramsearch 200
```

And for multi-class classification run

```bash
python baselines.py --method <method_name> --datadir <datadir> --logdir log --paramsearch 300 --multi
```

The results (cpmparison between different hyper-parameters) are saved
in `plot/baseline-[method_name]-model-[binary or multi].csv` 