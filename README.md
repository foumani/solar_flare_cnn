## Experimental Settings
The experiments were conducted on a system with an Intel(R) Xeon(R) Gold 5220R CPU, a NVIDIA RTX A5000 GPU, and 756 GB of RAM.
Please note that the RAM and GPU RAM requirements for this code is high (this is the case for the **Training** only) and program ram usage can go up to 128 GB and its GPU RAM usage
can go up to 16GB.

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

To reproduce the results presented in the paper, run the following command (replace <data_dir> with the path to the directory where the dataset partitions were extracted):

```bash
python experiments.py --datadir <data_dir> --experiment cmod --splitreport split.csv --modelreport model.csv --configreport config.csv
```
This command runs the main experiment using the CMOD model.


## Running Baseline Models
To run experiments for baseline models, use the following command:

```bash
python experiments.py --datadir <data_dir> --experiment <model_name>
```

Replace <model_name> with one of the following options:

* svm
* cif
* contreg
* cnn (refers to the CNN model implemented in the sktime library)
* lstm
* minirocket
* macnn

Each option runs the corresponding baseline model using the same dataset.