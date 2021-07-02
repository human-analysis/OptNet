# Adversarial Representation Learning with Closed-Form Solvers


### Requirements

1. Require `Python3`
2. Require `PyTorch1.0`
3. Require `Visdom0.1.8.9`
4. Check `requirements.txt` for detailed dependencies.

### Commands to Reproduce Results in Paper

#### Synthetic Gaussian Dataset
##### SGDA-ARL
~~~~
$ python3 -m visdom.server
$ python3 main_gaussian.py --args args/Gaussian-SGDA-ARL.txt
~~~~
##### OptNet-ARL
~~~~
$ python3 -m visdom.server
$ python3 main_gaussian.py --args args/Gaussian-OptNet-ARL.txt
~~~~
#### CelebA Dataset
##### SGDA-ARL
~~~~
$ python3 -m visdom.server
$ python3 main_celebA.py --args args/CelebA-SGDA-ARL.txt
~~~~
##### OptNet-ARL
~~~~
$ python3 -m visdom.server
$ python3 main_celebA.py --args args/CelebA-OptNet-ARL.txt
~~~~

#### Part A: Training the Encoder

1. Set the path to your input data and your dataset name for both training and test sets.
    **Note:** Let the data created by `dataloader.py` contain three items, input data, target class label
    and sensitive class label, respectively.
    Example in `args/CelebA-OptNet-ARL.txt`:
    ```
    dataset_root_test = ./data/celeba/
    dataset_root_train = ./data/celeba/
    dataroot = ./data/celeba/
    
    dataset_train = CelebA_Privacy
    dataset_test = CelebA_Privacy
    
    input_filename_train = ./data/celeba/celeba-training.csv
    input_filename_test = ./data/celeba/celeba-evaluation.csv

2. Set the dimentionality of your embedding `r` and data`ndim`, number of sensitive class label
    `nclasses_A`, and number of target class label `nclasses_T`.
    Example in `args.txt`:
    ```
    r = 2
   #### due to instant normalization one dimension will be lost
   
    resolution_high = 112
    resolution_wide = 96
    nclasses_A = 100
    nclasses_T = 20
    ```
3. Set a set trade-off parameters (0<=`alpha`<=1) between privacy and utility.
    **Note:** `alpha=[0]` is related to no privacy and `alpha=[1]` concerns totally
    to hide the sensitive attribute.
    ```
    alpha = [0, 0.1, 0.3 ,0.5, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94]

5. Choose your ARL method and associated networks.

    Example for  OptNet-ARL:
   
    ```
    adverserial_type = OptNet
    loss_type_E = Projection_gauss
    sigma = 1

   ```
    Example for  SGDA-ARL:
   
    ```
    adverserial_type = SGDA
    model_type_EA = EA
    model_type_ET = ET
   ```


#### Part B: Training the Real Adversary and Target Classifiers or Regressors

1. Visualization Settings.
The parameters for visdom to plot training and testing curves.

        1) the port number for visdom -- "port"
        2) the name for current environment -- "env"
        3) if you want to create a new environment every time you run the program or
         not -- "same_env".  If you do, set it "False"; otherwise, it's "True".

    Example in `args.txt`:
    ```
    port = 8097
    env = main
    same_env = True
    ```

2. Select the network for target and adversary and specify their task as a regression or classification.
Example in `args.txt`:
    ```
    model_type_A = Adversary
    model_type_T = Target
    loss_type_A = Regression
    loss_type_T = Regression
    evaluation_type_A = Top1Classification
    evaluation_type_T = Top1Classification
    ```

3. Finally, set the hyper parameters required to train and test the real adversary and target networks.
Example in `args.txt`:
    ```
    nepochs = 7
    optim_method = Adam
    learning_rate_T = 3e-4
    learning_rate_A = 3e-4
    scheduler_method_A = ExponentialLR
    scheduler_options_A = {"gamma": 0.999}
    ```