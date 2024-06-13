
# 1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness
This repository contains code for the paper 
[1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness](https://berndprach.github.io/publication/1LipschitzLayersCompared),
following an attempt to clean it up.
The original code can be found [here](https://github.com/berndprach/1LipschitzLayersCompared/)

**Work in progress!**

# Instructions:

Runs scrips e.g. in the following way:
```[bash]
python run.py scripts/step1_measure_batch_time.py 0
...
python run.py scripts/step1_measure_batch_time.py 31
```

```[bash]
python run.py scripts/step2_print_epoch_budgets.py
```


# Requirements:
 - PyTorch
 - torchvision
 - PyYAML
 - einops

