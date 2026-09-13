# Fashion-MNIST Neural Network 

This repository contains a Jupyter notebook implementing and analysing neural
networks on the Fashion-MNIST dataset, covering backpropagation from scratch,
activation functions, loss functions, optimisers, overfitting, regularisation,
and hyperparameter tuning with k-fold cross-validation.

## Contents

- `notebook.ipynb` — the full assignment notebook (Parts 1–7)
- `README.md` — this file

## Environment

- **Platform:** [Kaggle Notebooks](https://www.kaggle.com/)
- **Accelerator:** GPU T4 x2
- **Language:** Python 3.12
- **Main libraries:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`,
  `torch`, `tensorflow` (Keras)

All libraries used are pre-installed in the default Kaggle Python environment
— no additional installation is required when running on Kaggle.

## Dataset

- **Name:** Fashion-MNIST
- **Source:** https://www.kaggle.com/datasets/zalando-research/fashionmnist
- **Files used:** `fashion-mnist_train.csv`, `fashion-mnist_test.csv`

### To reproduce on Kaggle
1. Create a new Kaggle Notebook.
2. Under **Add Input**, attach the dataset
   `zalando-research/fashionmnist`.
3. Under **Settings → Accelerator**, select **GPU T4 x2**.
4. Upload/open `notebook.ipynb` and run all cells top to bottom.

### To reproduce locally (outside Kaggle)
1. Download the dataset CSVs from the Kaggle link above.
2. Update the file paths in the second cell (`pd.read_csv(...)`) to point to
   wherever you saved `fashion-mnist_train.csv` and `fashion-mnist_test.csv`
   locally.
3. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn torch tensorflow
   ```
4. Run the notebook top to bottom in Jupyter, JupyterLab, or VS Code.
   A GPU is recommended but not required — CPU will run correctly, just
   slower, particularly in Part 6 (regularisation study) and Part 7
   (k-fold hyperparameter search), which each involve many repeated
   training runs.

## How to run

Run all cells **in order, from top to bottom**. Later parts depend on
variables created in earlier parts:

| Part | Depends on |
|---|---|
| Part 1 (NumPy backprop) | Environment Setup cells |
| Part 2 (activation study) | Environment Setup cells |
| Part 3 (loss functions) | Part 2's `build_classifier`, one-hot labels |
| Part 4 (optimiser comparison) | Part 3's `build_classifier`, one-hot labels |
| Part 5 (forcing overfitting) | Environment Setup cells |
| Part 6 (regularisation study) | Part 5's `X_overfit`, `y_overfit_oh`, baseline results |
| Part 7 (hyperparameter tuning) | Part 6's regularisation results, test set (`test_data`) |

The real held-out test set (`fashion-mnist_test.csv`) is loaded only in
**Part 7**, consistent with the assignment instruction to keep it untouched
until then.

## Notes on reproducibility

- Random seeds are fixed throughout (`np.random.seed(42)`,
  `tf.random.set_seed(42)`, `random_state=42` in scikit-learn calls) to
  keep data splits and model initialisation consistent across runs.
- **GPU training is not fully bit-for-bit deterministic.** Operations such
  as Dropout, BatchNormalization, and data augmentation draw random numbers
  from TensorFlow's GPU kernels, which can execute in a slightly different
  order between runs. As a result, exact accuracy/loss numbers may vary by
  a small amount (typically 1–3 percentage points) between reruns, even
  with seeds set. The overall patterns and conclusions reported in the
  notebook are consistent across runs; only the precise decimal values may
  differ slightly.
- Part 1's from-scratch NumPy implementation is verified against PyTorch's
  autograd in the same section (gradient and loss differences on the order
  of 1e-7 to 1e-8, consistent with float32 rounding).

## Estimated runtime

Running the full notebook end-to-end on a Kaggle T4 GPU takes approximately
1–1.5 hours, with Part 6 (regularisation study, ~13 training runs) and
Part 7 (random search with 5-fold CV, ~60+ training runs) accounting for
most of that time.

## Report

Written answers, tables, and commentary required by the assignment
(explanations, the regularisation comparison, the final model evaluation,
etc.) are included as markdown cells and printed output directly within
`notebook.ipynb`.
