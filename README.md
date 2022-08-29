# OODGAT_KDD
This is the repository for paper 'Learning on Graphs with Out-of-Distribution Nodes'. 
Link: https://dl.acm.org/doi/abs/10.1145/3534678.3539457
To run the scripts, please create a folder named 'pyg_data' in the root directory. In the first run, datasets will be automatically downloaded to the folder from Pytorch_Geometric.
## Usage
### install dependencies
> pip install torch_geometric  
> pip install networkx  
> pip install sklearn  
> pip install numpy  
> pip install scipy  
### download datasets
(2022/8/30)
Recent versions of torch_geometric seems to have changed the default directory structure of automatically downloaded datasets, therefore the script may not work properly. In such case, please unzip the file 'pyg_data.zip' to the root directory which contains the older version of datasets.
### run baselines
> python baseline.py 
### run OODGAT
> python train.py
### test on different datasets
To test the models on different datasets, please change the corresponding parameters in train.py. For example, to change the dataset from 'Planetoid_Cora' to 'Amazon_Computers', simply modify the following lines:  
from  
> parser.add_argument('--dataset', default="Planetoid_Cora", type=str)

to  
> parser.add_argument('--dataset', default="Amazon_Computers", type=str)  

When dataset is changed, the 'splits', 'ID_classes' and 'continuous' should also be changed accordingly.  

| Dataset       | ID_classes           | Splits  |  Continuous  |
| :-------------: |:-------------:|:-----:|:-----:|
| Planetoid_Cora | [4, 2, 5, 6] | [20, 10, 1000] | False |
| Amazon_Computers | [8, 1, 2, 7, 6] | [20, 10, 5000] | False |
| Amazon_Photo | [3, 4, 5, 2, 0] | [20, 10, 3000] | False |
| Coauthor_CS | [5, 11, 10, 7, 14, 8, 12, 6] | [20, 10, 8000] | False |
| LastFMAsia | [0, 6, 14, 8, 15, 16, 11, 7, 13] | [20, 10, 3000] | True |
| wiki-CS | [3, 9, 7, 1, 8, 6] | [20, 10, 5000] | True |

### change hyperparameters
Hyperparameters for all datasets are listed in the following table:

| Dataset          | w_consistent | w_ent | w_discrepancy | margin | heads |
|:-----------------|:------------:|:-----:|:-------------:|:------:|:-----:|
| Planetoid_Cora   |      2       | 0.05  |     0.005     |  0.6   |   4   |
| Amazon_Computers |      2       | 0.05  |     0.005     |  0.4   |   4   |
| Amazon_Photo     |      3       | 0.10  |     0.005     |  0.4   |   4   |
| Coauthor_CS      |      4       | 0.05  |     0.005     |  0.6   |   4   |
| LastFMAsia       |      3       | 0.30  |     0.005     |  0.5   |   1   |
| wiki-CS          |      3       | 0.20  |     0.005     |  0.5   |   4   |
