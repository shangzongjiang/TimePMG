import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import LlamaConfig, LlamaModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from sklearn.preprocessing import StandardScaler


df = pd.read_csv('./dataset/ETThVis.csv')
# X=df[['HULL']].iloc[:8640]
X = df.iloc[:17419, :]
X_sample = X.sample(n=500, random_state=42)
# np.random.seed(1)
# selected_indices = np.random.choice(17420, size=5000, replace=False)

# embeddings_subset = df[selected_indices]

X = X_sample.select_dtypes(include=['float64', 'int64'])
# 3. 使用t-SNE进行降维（将数据降至2维用于可视化）
# tsne = TSNE(n_components=2, random_state=42)
# tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=500, random_state=42)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, n_iter=400, init='random', random_state=42, early_exaggeration=50, n_jobs=-1)
X_tsne = tsne.fit_transform(X)

# 4. 可视化t-SNE结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=10,c='red')
# plt.title('t-SNE Visualization')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.show()
# plt.axis('off')

plt.xticks([])  # 去掉 x 轴刻度
plt.yticks([])  # 去掉 y 轴刻度

# model_dir = "/mnt/external/szj/szj/GPT/GPT2_s/"
# llm_model = GPT2Model.from_pretrained(model_dir, output_attentions=True,
#                                            output_hidden_states=True)  # loads a pretrained GPT-2 base model
#
# word_embeddings = llm_model.get_input_embeddings().weight
#
# embeddings = word_embeddings.detach().cpu().numpy()  # 如果使用的是GPU，要确保将其移动到CPU并转换为numpy数组
# embeddings_subset = embeddings[:640]
# # 对数据进行标准化
# scaler = StandardScaler()
# embeddings_scaled = scaler.fit_transform(embeddings_subset)  # 标准化每个特征
#
# # 使用 PCA 降维到 50 维
# # pca = PCA(n_components=50)
# # embeddings_pca = pca.fit_transform(embeddings_scaled)
#
# # 使用 t-SNE 降维到 2 维
# tsne = TSNE(n_components=2, perplexity=30, learning_rate=500, n_iter=1000, random_state=42)
# # tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings_scaled)
#
# # 可视化降维后的嵌入
# plt.figure(figsize=(10, 10))
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
#
# plt.title('t-SNE Visualization of GPT-2 Word Embeddings (Optimized)')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')

# embeddings_subset = embeddings[:1000]
# # 使用 t-SNE 将嵌入降到2维
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings_subset)
#
# # 可视化降维后的嵌入
# plt.figure(figsize=(10, 10))
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
#
# plt.title('t-SNE Visualization of GPT-2 Word Embeddings (No Labels)')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
plt.savefig('./tsne_group_featuresV2.pdf')
plt.show()