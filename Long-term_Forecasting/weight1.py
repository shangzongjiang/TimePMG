import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler



# model = MyModel()  # 你的模型类
model.load_state_dict(torch.load('checkpoint.pth'))  # 加载训练好的权重
model.eval()