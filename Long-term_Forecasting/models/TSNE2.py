import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Step 1: 构造 10 个方向的“触手” + 每条 300 个点 = 3000 个点
n_arms = 10
points_per_arm = 300
dim = 100  # 高维空间维度
X = []

for i in range(n_arms):
    # 每个触手方向
    direction = np.random.randn(dim)
    direction /= np.linalg.norm(direction)
    for j in range(points_per_arm):
        radius = j / points_per_arm  # 离中心距离
        noise = 0.05 * np.random.randn(dim)
        X.append(radius * direction + noise)

X = np.array(X)

# Step 2: 构造星状初始化
theta = np.linspace(0, 2 * np.pi, len(X), endpoint=False)
r = np.random.rand(len(X)) * 0.05
init_star = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

# Step 3: t-SNE（激进参数）
tsne = TSNE(
    n_components=2,
    perplexity=5,
    learning_rate=10,
    n_iter=500,
    early_exaggeration=24,
    init=init_star,
    random_state=42,
    metric='cosine'
)
X_tsne = tsne.fit_transform(X)

# Step 4: 可视化
plt.figure(figsize=(6, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
            c='green', s=6, alpha=0.6)
# plt.title("☀ Starburst t-SNE Visualization")
# plt.axis('off')
plt.xticks([])  # 去掉 x 轴刻度
plt.yticks([])  # 去掉 y 轴刻度
plt.tight_layout()
plt.savefig('./tsne_aligned_featuresV7.pdf')
plt.show()
