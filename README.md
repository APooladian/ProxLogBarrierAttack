# ProxLogBarrierAttack
Public repository for the ProxLogBarrier attack, described in [A principled approach for generating adversarial images under non-smooth dissimilarity metrics](https://arxiv.org/abs/XXXX.XXXXX).

Abstract: Deep neural networks perform well on real world data, but are prone to
adversarial perturbations: small changes in the input easily lead to
misclassification. In this work, we propose an attack methodology catered not only for cases where the perturbations are measured by Lp norms, but in fact any adversarial dissimilarity metric with a closed proximal form. This includes, but is not limited to, L1,L2,L_inf perturbations, and the L0 counting "norm", i.e. true sparseness. Our approach is a natural extension of a recent adversarial attack method, and eliminates the differentiability requirement of the metric. We demonstrate our algorithm, ProxLogBarrier, on the MNIST, CIFAR10, and ImageNet-1k datasets. We consider undefended and defended models, and show that our algorithm transfers to various datasets with little parameter tuning. Furthermore, we observe that ProxLogBarrier obtains the best results with respect to a host of modern adversarial attacks specialized for the L0 case.

## Implementation details
Code is written in Python 3 and PyTorch 1.0. The implementation takes advantage of the GPU: a batch of images can be attacked at a given time. Hyperparameters are the defaults for the MNIST, CIFAR10, and ImageNet-1k datasets. If using Fashion-MNIST or CIFAR100, make sure you are using the default parameters and not the ones for ImageNet-1k. 

We provide a pre-trained MNIST model, with an example attack provided in `MNIST_example.py`. The attack is implemented as a class in `proxlogbarrier.py`. The code is simple enough that it should be transferrable to TensorFlow if necessary.

### Citation
If you find the ProxLogBarrier attack useful in your scientific work, please cite as
```
@article{pooladian2019proxlogbarrier,
  title={A principled approach for generating adversarial images under non-smooth dissimilarity metrics},
  author={Pooladian, Aram-Alexandre and Finlay, Chris and Hoheisel, Tim and Oberman, Adam M.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2019},
  url={http://arxiv.org/abs/XXXX.XXXXX},
  archivePrefix={arXiv},
  eprint={XXXX.XXXXX},
}
```
