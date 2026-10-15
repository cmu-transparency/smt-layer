# SMT Layer

**Code coming soon!**

Implementation of SMTLayer in Pytorch with Z3
This code is the python companion to **Grounding Neural Inference with Satisfiability Modulo Theories**, which is appearing in NeurIPS 2023.

 In this paper we present a set of techniques for integrating Satisfiability Modulo Theories (SMT) solvers into the forward and backward passes of a deep network layer, called SMTLayer. **Notably, the solver needs not be differentiable.** We implement SMTLayer as a Pytorch module. An overview of our work is shown as follows.

<img width="1026" alt="smt_layer" src="./images/teaser_figure.png">

SMTLayers, when used on top of other neural network layers, can be leveraged to solve many tasks requiring logical reasoning. For example, the addition of two digits in an image. Morever, we show how to do visual Sudoku, Lier's Puzzle (above), etc.
<img width="660" alt="example" src="./images/mnist_addition.png">


We implement SMTLayer as a Pytorch module, and our empirical results show that it leads to models that 1) require fewer training samples than conventional models, 2) that are robust to certain types of covariate shift, and 3) that ultimately learn representations that are consistent with symbolic knowledge, and thus naturally interpretable.

<img width="1000" alt="table" src="./images/table.png">


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


## Matched MNIST addition verification

Run both trained checkpoints with the same CROWN method, pixel boxes, seeded
test pairs, optimization budget, CPU thread count, and per-example timeout:

```bash
uv run python scripts/verify_mnist_addition.py \
  --checkpoint checkpoints/mnist_addition_baseline_pct100_seed0_trial1.pt \
               checkpoints/mnist_addition_smt_pct100_seed0_trial1.pt \
  --examples 5 --sample-seed 0 \
  --eps 0 0.003921568627 0.007843137255 0.01568627451 0.03137254902 \
  --bound-method CROWN-Optimized \
  --alpha-iterations 20 --alpha-max-time 30 --example-timeout 180 \
  --output matched_verification.csv
```

Both models use auto_LiRPA; neither uses branch-and-bound or an attack in this
comparison. The baseline certifies its five output-bit signs. The SMT model
bounds eight operand logits and checks all allowed bit assignments using exact
binary addition with operands 0–15, matching the model's theory. Zero is True.
`inconclusive` is not evidence of an adversarial example.

Pairs are shuffled with a local seed; every block of 100 covers all ordered
digit-pair types. Images are sampled from MNIST's test split and their dataset
indices are saved. This differs from the old first-image, lexicographic pilot.

Results are saved after every model/example/epsilon task:

- `matched_verification.csv`: summaries, completed counts, outcomes, runtime.
- `matched_verification_examples.csv`: per-example outcomes, bounds, settings,
  image indices, runtime (including worker startup), and bound-computation time.
- `matched_verification.resume.json`: durable task state and run identity.
- `matched_verification_logs/`: library diagnostics for each task.

Rerun the identical command to resume; completed tasks, including timeouts and
errors, are skipped. An interrupted active task is rerun. Settings, checkpoint
hashes, sample identities, relevant package versions, and script hash must match.
Use a new output filename to change settings or retry timed-out tasks. Do not
run concurrent jobs with the same output path. Existing pilot files are not
overwritten or imported into the new comparison.

`--alpha-max-time` is a soft limit per bound direction. `--example-timeout` stops
the worker at the wall-clock limit, including initialization; cleanup can take
a few extra seconds. Timeouts and errors never count as certificates. Summaries
during an unfinished run cover only completed tasks; check `completed_examples`.
`clean_accuracy` covers tasks whose clean prediction was obtained, with
`clean_evaluated` recording that denominator. At epsilon zero the exact clean
prediction is sufficient and optimization is skipped for both models.

For a faster, looser matched first pass, use `--bound-method CROWN` with a new
output filename. For a representative experiment, increase `--examples` to
100 or more after the pilot succeeds.

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
