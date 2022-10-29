# Sample Code for Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```


## Download Model
```shell
# To download model from dropbox
bash download.sh
```

## Intent Inference
```shell
bash intent_cls.sh {file/to/test.json} {file/to/save/prediction_intent.csv}
```

## Slot Inference
```shell
bash slot_tag.sh {file/to/test.json} {file/to/save/prediction_slot.csv}
```
