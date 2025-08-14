import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Step 1: 本地路径
local_model_path = "/mnt/external/szj/szj/GPT/GPT2_s/"

# Step 2: 从本地加载模型和 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
model = GPT2Model.from_pretrained(local_model_path)

# Step 3: 提取 embedding 权重 [vocab_size, hidden_size]
embedding = model.transformer.wte.weight.detach().cpu().numpy()

# Step 4: 随机采样 500 个 token 向量
np.random.seed(42)
indices = np.random.choice(embedding.shape[0], size=500, replace=False)
subset_embeddings = embedding[indices]
subset_tokens = [tokenizer.decode([i]) for i in indices]

# Step 5: PCA 降维到 50 维，再 t-SNE 到 2D
pca = PCA(n_components=50, random_state=42)
subset_pca = pca.fit_transform(subset_embeddings)

tsne = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42)
subset_tsne = tsne.fit_transform(subset_pca)

# Step 6: 可视化
plt.figure(figsize=(10, 8))
plt.scatter(subset_tsne[:, 0], subset_tsne[:, 1], s=12, alpha=0.6, c='orange')

# （可选）加部分 token 标注
for i, tok in enumerate(subset_tokens):
    if np.random.rand() < 0.1:
        plt.text(subset_tsne[i, 0], subset_tsne[i, 1], tok, fontsize=7, alpha=0.7)

plt.title("t-SNE of Local GPT Token Embeddings", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
