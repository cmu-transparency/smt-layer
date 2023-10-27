# SMT Layer

**Code coming soon!**

Implementation of SMTLayer in Pytorch with Z3
This code is the python companion to **Grounding Neural Inference with Satisfiability Modulo Theories**, which is appearing in NeurIPS 2023.

 In this paper we present a set of techniques for integrating Satisfiability Modulo Theories (SMT) solvers into the forward and backward passes of a deep network layer, called SMTLayer. **Notably, the solver needs not be differentiable.** We implement SMTLayer as a Pytorch module. An overview of our work is shown as follows.

<img width="1026" alt="smt_layer" src="https://github.com/cmu-transparency/smt-layer/assets/9357853/35ae93dc-af5d-4f91-82de-518dd7434faa">

SMTLayers, when used on top of other neural network layers, can be leveraged to solve many tasks requiring logical reasoning. For example, the addition of two digits in an image. Morever, we show how to do visual Sudoku, Lier's Puzzle (above), etc.
<img width="660" alt="example" src="https://github.com/cmu-transparency/smt-layer/assets/9357853/ac36a246-9322-41ef-9096-cdd2916e76ee">


We implement SMTLayer as a Pytorch module, and our empirical results show that it leads to models that 1) require fewer training samples than conventional models, 2) that are robust to certain types of covariate shift, and 3) that ultimately learn representations that are consistent with symbolic knowledge, and thus naturally interpretable.

<img width="1000" alt="table" src="https://github.com/cmu-transparency/smt-layer/assets/9357853/4afe0cf3-043f-4945-902f-ee600a38d23b">


## Prerequisites

You may need the following dependencies installed on your system:

- Python (3.6+)
- PyTorch
- torchvision
- matplotlib
- livelossplot
- z3

Depending on your operating system, you can install z3 using different methods. Here's how to install z3 using `pip`:

```bash
pip install z3-solver
```

Otherwise follow the instructions [here](https://github.com/Z3Prover/z3).


## Data Preparation
Datasets used in the paper are included in `notebook/` execept MNIST, which is publicly available. For example, you can download from Pytorch.
```python
import torchvision
from torchvision import transforms

# Define a transformation
transform = transforms.Compose([transforms.ToTensor()])

# Download and transform the training dataset
mnist_train = torchvision.datasets.MNIST(
    '/data/data', 
    train=True, 
    download=True, 
    transform=transform
)

# Download and transform the test dataset
mnist_test = torchvision.datasets.MNIST(
    '/data/data', 
    train=False, 
    download=True, 
    transform=transform
)
```

Replace '/data/data' with the appropriate path where you want to store the dataset.


## Acknowledgments

If you use this code please cite
```
@inproceedings{
    wang2023grounding,
    title={Grounding Neural Inference with Satisfiability Modulo Theories},
    author={Zifan Wang and Saranya Vijaykumar and Kaiji Lu and Vijay Ganesh and Somesh Jha and Matt Fredrikson},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=r8snfquzs3}
}
```
