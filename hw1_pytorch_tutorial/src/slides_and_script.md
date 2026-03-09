# HW1 Video Submission — Script (2 slides, ~60 seconds)

---

# Script

## [0-15s] Slide 1: Exercise 3

"For Exercise 3, I built a 3-layer MLP classifier for MNIST with GELU activations, trained using Adam at a learning rate of 1e-3. As you can see from the training loss curve on the left, the loss drops sharply in the first epoch and continues to decrease. The test loss on the right also steadily decreases across all 5 epochs. The model reaches about 97% test accuracy, well above the 70% threshold."

## [15-35s] Slide 2: Exercise 4 — Setup & Results

"Moving to Exercise 4. I implemented a tiny Vision Transformer for MNIST — it splits each image into 4-by-4 patches, giving 49 tokens, then applies 2 Transformer encoder blocks. I compared three FFN variants: the standard baseline with GELU, GEGLU, and SwiGLU. I chose these two because they were the best performers in Shazeer's GLU Variants paper. To keep it fair, GLU variants use the two-thirds width rule, so all three have roughly 105,000 parameters — the difference is less than 0.1%."

## [35-50s] Slide 2: Observations

"Looking at the plots, both GLU variants clearly outperform the baseline. GEGLU reaches 96.9% versus the baseline's 95.1%. You can also see faster convergence — at epoch 1, GEGLU is already 2 points ahead. The gating mechanism lets the model learn which features to suppress by multiplying two separate projections element-wise, which a standard activation cannot do. The ranking — GEGLU, then SwiGLU, then baseline — matches Table 1 from the paper."

## [50-60s] Slide 2: Reproducibility

"On reproducibility: I used a fixed seed for initialization, but with only one run per variant, these gaps aren't confirmed as statistically significant. Running 3 to 5 seeds would be needed. The effect also looks amplified at this tiny scale compared to the paper's large-scale experiments."
