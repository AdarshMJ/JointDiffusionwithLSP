#### Instructions on how to run the code ####

#### Requirements ####
1. Python 3.11 or higher
2. Pytorch
3. Pytorch Geometric
4. Networkx

## Since the datasets are too large to be zipped. The CoraChameleon datasets need to be generated ##

## Generating training data
```python
python graph_generator.py --output-dir data --node-sizes 100 --train-per-set 3000 --test-count 100 --visualize --real-world --s1-source-datasets Cora Chameleon
```

## Training regular DiGress

```python
python main.py --config configs/configfilename.yaml
```


## Training regular JointLSP+DiGress
```python
python main_lsp.py --config configs/configforlsp.yaml
```