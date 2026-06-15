# AdaBoost: ML Classification from Scratch

This project implements the AdaBoost ensemble learning algorithm from scratch
using Python and NumPy. The goal is to evaluate how boosting improves
classification performance compared to a single decision tree (decision stump
baseline) on a real-world marketing dataset.

The dataset is derived from direct marketing campaigns (phone calls) conducted
by a Portuguese banking institution. The classification objective is to predict
whether a client will subscribe to a term deposit (`y`).

## What's implemented from scratch

- The full AdaBoost boosting loop: weighted training of decision stumps,
  weighted-error computation, vote-weight (alpha) calculation, and exponential
  re-weighting of misclassified examples after each round.
- Preprocessing of the raw dataset: parsing the semicolon-delimited file,
  encoding categorical columns numerically, and a stratified train/test split.

scikit-learn is used only for the individual decision stumps; the boosting
logic, weighting, and final ensemble prediction are implemented by hand.

## Results

| Metric | Value |
|---|---|
| AdaBoost test accuracy | 89.7% |
| AdaBoost training accuracy | 89.9% |

The boosted ensemble outperforms the single decision stump baseline (shown in
the plot below), and the close gap between training and test accuracy indicates
the model generalizes well, with no significant overfitting.

Because only ~12% of clients subscribe to a term deposit, a naive majority-class
predictor would already reach roughly 88% accuracy, so the model is compared
directly against the single decision stump rather than on raw accuracy alone.

![Training and testing error across boosting rounds](https://github.com/user-attachments/assets/85f1e466-0fe9-446b-b202-9e7da93fd486)

The green dashed line (test error) falls below the red dashed line (single
decision tree error), confirming that boosting improves on the baseline. The
blue line is the unweighted error of each individual stump — its spikiness is
expected, since AdaBoost trains each new stump on a re-weighted dataset that
emphasizes the examples the ensemble currently misclassifies, so an individual
stump can look poor on the original unweighted data while still strengthening
the ensemble.

## Running it

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook adaboost.ipynb
```

Then run all cells. The Bank Marketing dataset (`bank-full.csv`) must be present
so the notebook can load it — download it from the UCI link below if it isn't
already in the repo.

## Repository structure

- `adaboost/` – Jupyter notebook implementing AdaBoost from scratch
- `dataset/` – the Bank Marketing dataset used for training and testing

## Dataset

UCI Bank Marketing Dataset — https://archive.ics.uci.edu/ml/datasets/bank+marketing
(Moro et al., 2014)
