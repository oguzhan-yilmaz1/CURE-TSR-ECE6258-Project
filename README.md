
DIRECTORY NOTE
In order for the existing script to work, we need to link in some data in a 
particular structure.

Create a directory like follows:

    data/
        RealChallengeFree/
            train=(symlink to some training data)
            Test=(symlink to some testing data)

### Requirements
- Tested on Linux 14.04
- CUDA, CuDNN
- Anaconda (or virtualenv)
- PyTorch (www.pytorch.org)
- Optionally, tensorflow-cpu for tensorboard


### Alternative download for Dataset
wget "https://ieee-dataport.s3.amazonaws.com/open/708/CURE-TSR.zip?response-content-disposition=attachment%3B%20filename%3D%22CURE-TSR.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20201201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201201T063617Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=05e3312a718088957c29dd8043558c28333b9de14974263bdc50ea76c734f7aa"

### Usage

```
usage: train.py [-h] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR]
                [--momentum M] [--weight-decay W] [--print-freq N]
                [--resume PATH] [-e] [--test model_dir] [--net MODEL] [--loss LOSS]
                DIR

CURE-TSR Training and Evaluation

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  -- test		test the model after complete training using saved checkpoint
  -- net		Select one of 'rgb','intensity','cnn','sift','autoencoder'
  -- loss		Select one of 'softmax','svm'
```

- Training example:
```
python train.py --lr 0.001 --net rgb --loss svm ./data
```
- Testing example: You need to change the variable 'testdir' to test trained models on different challenging conditions. 


- Testing example command
```
python train.py --net rgb --svm --test ./checkpoints/RGB-SVM/model_best.pth.tar ./data

Using the net and loss variants of the above command, you can check all the reported results in this report as well as obtain approximately the same values for original paper's baseline performance for all challenge types.

