
# 1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness
This repository contains code for the paper 
[1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness](https://berndprach.github.io/publication/1LipschitzLayersCompared),
following an attempt to simplify it and clean it up.
The original code can be found [here](https://github.com/berndprach/1LipschitzLayersCompared/)

<img src="https://github.com/berndprach/OneLipschitzLayersCompared/blob/main/data/star_plot.png" alt="Radar plot of results" width="800"/>

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

```[bash]
python run.py scripts/step3_hp_search.py 1
```

# Reproduce Experiments from the Paper:
Follow the pseudocode below:
```{r, tidy=FALSE, eval=FALSE, highlight=FALSE }
For i = 0 to 31 
    >> python run.py scripts/step1_measure_batch_time.py i
    
>> python run.py scripts/step2_print_epoch_budgets.py
Update data/epoch_budgets_2h/cifar10.yaml with the printed values

For i = 0 to 31
    For _ = 1 to 16
        >> python run.py scripts/step3_hp_search.py i

>> python run.py scripts/step4_print_best_hp.py
Update data/best_hps/cifar10.yaml with the printed values

For i = 0 to 31
    >> python run.py scripts/step5_test_set_evaluation.py i
```
Repeat for the other datasets and other time budgets.

# Requirements:
 - PyTorch
 - torchvision
 - PyYAML
 - einops

