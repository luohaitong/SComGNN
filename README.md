#  Spectral-based Graph Neural Networks for Complementary Item (AAAI 2024 Main Track)


## Dataset Preprocessing
- Step1: Download meta data from https://nijianmo.github.io/amazon/index.html.
- Step2: Put the meta data file in <tt>./data_preprocess/raw_data/</tt>.
- Step3: Set the dataset name (i.e., <tt>$dataset</tt>) in run.sh, run preprocessing by 
    ```
    cd data_preprocess
    sh run.sh
    ```
## Running Experiments
For the integration of low-frequency information and mid-frequency information, we provide four versions corresponding to the ablation study part of the paper.

Example of training SComGNN on Appliances dataset:
```
python run.py
```
Example of training SComGNN (w/o a) on Toys dataset:
```
python run.py --dataset Toys --lr 0.01 --mode concat
```
Example of training SComGNN (w/o m) on Grocery dataset:
```
python run.py --dataset Grocery --lr 0.04 --mode low
```
Example of training ScomGNN (w/o l) on Home dataset:
```
python run.py --dataset Home --lr 0.04 --mode mid
```
