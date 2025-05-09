# training-Neural-networks-with-Keras---advnced
# 🧠 Deep Learning for Model Robustness & Keras Customization

This repository presents a collection of hands-on notebooks focusing on improving deep learning model robustness through **regularization techniques** and advanced **Keras customization**, including custom training loops, layers, and losses.

---

## 📁 Part 1: Robust Regularization Techniques in Keras

Explore a wide range of regularization strategies to combat overfitting and enhance model generalization across domains including vision, tabular, and multimodal data.

| Notebook | Description |
|----------|-------------|
| [`01_regularization_variants.ipynb`](https://colab.research.google.com/drive/1J6QuxUER7MO6KnkBDGPJ3Ty7Xo2lyCNd?usp=sharing) | Implements and compares L1, L2, and ElasticNet-style regularization using Keras layers. |
| https://colab.research.google.com/drive/1QkjK12Ri4xEhoorYTGNsCYo83cUEmCIO?usp=sharing | Demonstrates dropout techniques within ResNet-like architectures. |
| https://colab.research.google.com/drive/1OQekU6cfLDFRaVQKuTr8J0qDHIyvPG7d?usp=sharing | Applies Monte Carlo Dropout to quantify uncertainty in predictions. |
| https://colab.research.google.com/drive/1VuDz2cF3t_EfseEbW5Cta4BrzK4abPZO?usp=sharing | Compares EarlyStopping and ReduceLROnPlateau callbacks. |
| https://colab.research.google.com/drive/1rCxhFGHguOGvs2hlsQ1mYDfQ97OMlYnu?usp=sharing | Evaluates different initializers: He Normal, Orthogonal, and custom-defined. |
| https://colab.research.google.com/drive/160VqTa8mglqERBCpFQ_3Yxcd138hoioa?usp=sharing | Experiments with BatchNorm, LayerNorm, and GroupNorm for stable training. |
| https://colab.research.google.com/drive/1ejO3Om9iIXES6Zui1LbLd4505Yl5883S?usp=sharing | Defines and integrates a cosine-based custom regularization function. |
| https://colab.research.google.com/drive/10TSYvmYP5HPqw64r2KRZyXZHq4SybnJz?usp=sharing | Uses WandB/MLflow for real-time experiment tracking and visualization. |
| https://colab.research.google.com/drive/1AYOik7ZJvtsSNj0jGvDyCW6KPqB4lqfv?usp=sharing | Applies data augmentation using TensorFlow Addons. |
| https://colab.research.google.com/drive/1wOhMm1x5qlHnSapsU2rMcf6nRIB05ADj?usp=sharing | Uses Meta’s AugLy for augmenting text, video, and images. |
| https://colab.research.google.com/drive/1Uoi_EF3-sjs6kJkk4XT-3OefjxOfrBu9?usp=sharing | Demonstrates noise injection and SMOTE-style augmentation for tabular data. |
| https://colab.research.google.com/drive/1jscfodNE3FTQI3WhBRnp6oLrPlcuFKR5?usp=sharing | Comparative study of FastAI vs Keras augmentation pipelines. |

🎥 _ https://www.youtube.com/watch?v=IWTH0s75x7c

---

## 📁 Part 2: Advanced Keras – Custom Layers, Losses, Metrics & Training

Deep-dive into the internals of Keras by creating fully customized layers, activations, optimizers, and training loops.

| Notebook | Description |
|----------|-------------|
| https://colab.research.google.com/drive/12WiJx3fF4fV9vAJ_K3k95Ln_N3MihBZj?usp=sharing | Covers OneCycleLR, MC Dropout Alpha, custom normalization layers, advanced logging (WandB), custom loss/metrics, attention blocks (SE/CBAM), contrastive learning, and a flexible training loop implementation. |
Youtube Video : https://www.youtube.com/watch?v=IWTH0s75x7c
---

## ✅ How to Use This Repository

1. **Open any notebook in Google Colab** by clicking the badge or using `Open in Colab` links.
2. **Run all cells sequentially** – dependencies are auto-installed.
3. **Watch the video walkthrough** (where provided) for better understanding.
4. **Fork and remix the notebooks** for your own datasets and experiments.

---

## 📦 Dependencies

- `tensorflow`, `keras`
- `tensorflow-addons`
- `wandb` or `mlflow`
- `augly`
- `scikit-learn`, `matplotlib`, `pandas`
- `fastai` (for comparative augmentation)

Use the following to install everything:

```bash
pip install -r requirements.txt
