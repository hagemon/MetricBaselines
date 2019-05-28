# Metric Baselines

Implementations of some popular metric learning methods via PyTorch, including:

- Semi-hard Triplet Loss [1]
- N-pair Loss [2]
- Lifted Structured Loss [3]
- Angular Loss [4]
- Ranked List Loss [5]

We use GoogLeNet (Inception V1) [6] as our backbone of feature extractor.

We follow the settings mentioned in the papers of these works,
except using Euclidean distance in Lifted Structed Loss.

We may add more methods in the future.

If you find this project useful, feel free to cite methods above in your paper and star the referenced repos (mentioned in losses section).

Issues are welcomed!

## Code Architecture

### dataset.py

Implementations of Dataset class and interfaces of getting dataset and data loader.

### model.py

Build our network with Inception V1 [6] and randomly initialized fully connected layer, and provide a interface to get our model.

### sampler.py

We have two sampler here

- **Balanced Sampler** that pick a fix number of instances from classes.

- **Class Mining Sampler** that first random pick a class, then chose classes that close to the chosen one.

### test.py

Implementations of experiment metric including:

- R@1 R@2 R@4 R@8
- NMI
- F1 Score

### train.py

Just train.

**Remember change path to fit your environment.**

### transforms.py

Some pre-process to data.

### basic_model

Implementations of Inception v1, written by [TheCodeZ](https://github.com/TheCodez/vision/releases/tag/1.0).

### losses

Implementations of losses mentioned above.

Some of them are collected from [bnulihaixia](https://github.com/bnulihaixia/Deep_metric) and [adambielski](https://github.com/adambielski/siamese-triplet).

## Usage
### Datasets

The default dataset is CUB, you can change it by

```bash
python train.py --dataset $DATASET
```

We place the image path and label id for each sample in one line in **train.txt** and **test.txt**.
Here is a demo of train.txt

```
path/to/images/hello_kitty.jpg 0
path/to/images/Garfield.jpg 0
path/to/images/Snoopy.jpg 1
path/to/images/Goofy.jpg 1
...
```

### Args

- **--lr**: learning rate, default 0.00001.
- **--iteration**: training iteration, default 20000.
- **--dim** dimension of embedding, default 512.
- **--batch-size**: default 128.
- **--dataset**: default CUB.
- **-e**: evaluation model without training if used.
- **--method**: default Triplet
- **--balanced**: use balanced sampler if used.
- **--instances**: number of instances for each class, default 4.
- **--cm**: use class mining sampler if used, not conflict with balanced flag.

### Train the model

```bash
# For Semi-hard Triplet
python train.py --method Triplet

# For N-pair Loss (with class mining)
python train.py --method N_pair --balanced --instances 2 --batch-size 120 --cm

# For Lifted Structured Loss
python train.py --method Lifted --batch-size 120

# For Angular Loss (with n-pair selection)
python train.py --method Angular --balanced --instances 2 --batch-size 120

# For Ranked List Loss
python -u train.py --method RankedList --balanced --instances 3 --batch-size 180

```

### Test model

just add `-e` for each training command, our script
will automatically find trained model if exists.

## Results

| Method          | R@1    | R@2    | R@4    | R@8    | NMI    | F1     |
| --------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Semi-hard       | 0.3621 | 0.4693 | 0.5930 | 0.7108 | 0.4880 | 0.2807 |
| N-pair          | 0.4147 | 0.5418 | 0.6634 | 0.7714 | 0.5509 | 0.3475 |
| Lifted          | 0.4932 | 0.6120 | 0.7265 | 0.8256 | 05691  | 0.3902 |
| Angular(N-pair) | 0.4799 | 0.6041 | 0.7268 | 0.8210 | 05827  | 0.4136 |
| Rank List       | 0.5712 | 0.6877 | 0.7873 | 0.8661 | 0.6302 | 0.4647 |

## References

[1] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 815-823).

[2] Sohn, K. (2016). Improved deep metric learning with multi-class n-pair loss objective. In Advances in Neural Information Processing Systems (pp. 1857-1865).

[3] Oh Song, H., Xiang, Y., Jegelka, S., & Savarese, S. (2016). Deep metric learning via lifted structured feature embedding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4004-4012).

[4] Wang, J., Zhou, F., Wen, S., Liu, X., & Lin, Y. (2017). Deep metric learning with angular loss. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2593-2601).

[5] https://arxiv.org/pdf/1903.03238.pdf

[6] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
