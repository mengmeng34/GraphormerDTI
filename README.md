# GraphormerDTI
A graph transformer-based approach for drug-target interaction prediction.
## Requirements and Installation
```
pip install -r requirements.txt
```
## Preprocess
Under transductive setting, please run the code
```
python preprocess_transductive_setting.py
```
Under drug inductive setting, please run the code
```
python preprocess_drug_inductive_setting.py
```
Under drug-protein inductive setting, please run the code
```
python preprocess_drug_protein_inductive_setting.py
```
Using the Davis dataset under transductive setting as an example, the preprocessing might output the following log
```
Train in Davis
proteins load finished
data shuffle
10157 2540 12696
10157 finished
2540 finished
12696 finished
train dataset finished
valid dataset finished
test dataset finished
10157 2540 12696
```
The logs show that under the current random seed, the training set, validation set and test set include 10157, 2540 and 12696 drug-protein pairs, respectively. The pre-processing result is stored in ./molecule_data/Davis_transductive_setting_SEED.pkl.
## Run
Under transductive setting, please run the code
```
python main_transductive_setting.py
```
Under drug inductive setting, please run the code
```
python main_drug_inductive_setting.py
```
Under drug-protein inductive setting, please run the code
```
python main_drug_protein_inductive_setting.py
```
