# Experimental protocol and interpretation

## Hypothesis
With coupled Adam + L2, adding λθ to the gradient modifies both first and second moments;
with AdamW, λθ is applied separately as multiplicative shrinkage. This **does not imply**
that a numerically identical λ gives matched effective regularization or that AdamW
will always win on every dataset.

## Experimental tracks

| Track | Dataset | Network | Primary figures |
|---|---|---|---|
| A | deterministic anisotropic quadratic | two trainable scalar coordinates | loss contours and optimizer paths |
| B | sklearn Two Moons: 160/200/2000 train/validation/test | 2–64–64–1 Tanh MLP | decision surface, train/val BCE, norm and update ratio |
| C | torchvision CIFAR-10: 40k/10k/official 10k | CIFAR ResNet-18 (3x3 stem, no maxpool) | CE/accuracy, weight norm |
| D | Hugging Face WikiText-2 raw | 2-layer causal Transformer, GPT-2 tokenizer only | token CE, perplexity, weight norm |

## Comparison validity

- **Paired initialization:** same model init for every optimizer within each seed.
- **Paired data:** shared train/validation/test partition; reproducible mini-batch
  choices for Moons and language modeling. CIFAR loader and augmentation are
  seeded identically per optimizer with num_workers=0.
- **Same decay mask:** weights with ndim>=2 are decayed; biases and vector
  normalization weights are not. Quadratic explicitly decays its toy vector.
- **Matched base optimizer:** same betas and eps within each task. Adam baseline
  uses zero decay. Adam + L2 uses PyTorch Adam coupled weight_decay; AdamW uses
  decoupled weight_decay.
- **Task loss only:** logged losses are always *unregularized* task metrics
  for apples-to-apples comparison.
- **Validation for selection:** only validation data is used for hyperparameter
  sweeps. Official test data is measured at completion (not for selection).
- **Replicates:** run at least 5 paired seeds for a publication-grade result.
  Use paired distributions / confidence intervals; do not imply significance
  from a single seed.
- **Learning-rate interaction:** sweep learning rate and decay for each optimizer
  with comparable compute budgets and select on validation data.
- **Different λ are not equivalent:** equivalent nominal values do not guarantee
  equal effective shrinkage. Compare both matched nominal settings and
  independently tuned best runs.
- **CIFAR limitation:** train CE is measured on augmented mini-batches,
  while validation CE uses deterministic transforms; they are not identical
  distributions.
- **WikiText limitation:** train loss is current minibatch token CE;
  validation/test CE uses a **fixed prefix subset** of chunks capped by
  --max-eval-batches. Perplexity is exp(token CE), NOT word-level perplexity.
- **Performance:** logging parameter update norm for every parameter requires
  extra allocations; dense update-ratio monitoring is implemented only on
  Two Moons. CIFAR uses per-epoch weight norms.
- **Data leakage:** no validation/test samples used to fit the Two Moons scaler.
  CIFAR uses predefined published CIFAR-10 mean/std for normalization.
- **Caution:** multiple seeds and robust hyperparameter sweeps are required
  before asserting better generalization.
