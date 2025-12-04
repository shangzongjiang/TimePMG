import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_Freq_FourierInterpolate, DataEmbedding_FreqInterpolate, DataEmbedding_FreqComplex
from layers.FilterLayer import FrequencyDomainFilterLayer
from layers.Complex_Func import ComplexLayerNorm
from layers.Layers import Bottleneck_Construct

from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import LlamaConfig, LlamaModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import math
import torch.nn.functional as F
import numpy as np

from collections import defaultdict

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class ComplexProjection(nn.Module):
    def __init__(self, d_model, freq_len):
        super(ComplexProjection, self).__init__()
        self.linear_real = nn.Linear(d_model, d_model)
        self.linear_imag = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model * 2, freq_len)

    def forward(self, x):
        real_part = self.linear_real(x.real) - self.linear_imag(x.imag)
        imag_part = self.linear_imag(x.real) + self.linear_real(x.imag)
        x = torch.cat((real_part, imag_part), dim=-1)
        x = self.linear_out(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_out = configs.c_out
        self.c_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_norm = configs.use_norm
        self.filter_type = configs.filter_type
        self.quantile = configs.quantile
        self.bandwidth = configs.bandwidth
        self.embedding = configs.embedding # "fourier_interpolate" or "interpolate" or "complex"
        self.top_K_static_freqs = configs.top_K_static_freqs
        self.static_freq_indices = None
        self.model = nn.ModuleList([FrequencyDomainFilterLayer(
            self.seq_len, self.d_model, self.c_in,
            filter_type=configs.filter_type,
            bandwidth=self.bandwidth,
            top_K_static_freqs=self.top_K_static_freqs,
            quantile=self.quantile)
            for _ in range(configs.e_layers)])

        if self.embedding == "fourier_interpolate":
            self.enc_embedding = DataEmbedding_Freq_FourierInterpolate(self.seq_len, self.d_model, self.c_in)
        elif self.embedding == "interpolate":
            self.enc_embedding = DataEmbedding_FreqInterpolate(self.seq_len, self.d_model)
        else:
            self.enc_embedding = DataEmbedding_FreqComplex(self.seq_len, self.d_model)
        self.layer = configs.e_layers
        self.layer_norm = ComplexLayerNorm(self.d_model)
        self.projection = ComplexProjection(self.d_model, configs.pred_len)
        self.top_period = getattr(configs, 'top_period', [1])
        self.top_amp = getattr(configs, 'topk_amplitudes', [1]).long()

        print(self.top_period)


        ###using GPT
        self.is_gpt=configs.is_gpt
        if configs.llm_model == 'GPT2':
            if configs.pretrain:
                model_dir = "/mnt/external/szj/szj/GPT/GPT2_s/"
                self.llm_model = GPT2Model.from_pretrained(model_dir, output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.llm_model = GPT2Model(GPT2Config())
            self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.llm_model))



        ###multi-scale world token embeddings
        self.batch_size=configs.batch_size
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]





        self.num_tokens = 1000
        self.dim_mapping=nn.Linear(configs.enc_in,self.word_embeddings.shape[1])
        self.word_size=configs.word_size
        # word_all = torch.cat([self.vocab_size], self.word_size)
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.multi_word_layer=nn.ModuleList()
        self.hyperatten=nn.ModuleList()
        self.dec_alllen=0
        for i in range(len(self.word_size)):
            ###计算总的节点数
            if (self.seq_len)% self.top_period[i]!=0:
                length = ((self.seq_len) // self.top_period[i]) + 1
            else:
                length=self.seq_len // self.top_period[i]
            self.dec_alllen=length+self.dec_alllen+self.word_size[i]
            # self.hyperatten.append(HyperAtten(self.word_embeddings.shape[1],configs.enc_in))
            self.hyperatten.append(HyperAtten(self.word_embeddings.shape[1], self.word_embeddings.shape[1]))
            if i!=0:
                self.multi_word_layer.append(nn.Linear(self.word_size[i-1], self.word_size[i]))
                # word_all.append()
            else:
                self.multi_word_layer.append(nn.Linear(self.vocab_size, self.word_size[i]))
        # print("okk")

            """
            if (self.seq_len)% period != 0:
                length = (((self.seq_len) // period) + 1) * period
                # padding = torch.zeros((length - self.seq_len))
                # out = torch.cat([torch.tensor([self.seq_len]), padding])
                # out = torch.cat([
                #     torch.tensor([self.seq_len], dtype=padding.dtype, device=padding.device),
                #     padding
                # ])
                all_out.append(length)
            else:
                length = (self.seq_len)
                out=length // period
                # out = self.seq_len
                all_out.append(out)
            """


        self.multi_conv=Bottleneck_Construct(self.word_embeddings.shape[1],self.word_embeddings.shape[1],self.top_period)
        ###high-order CMA
        self.incidence = incidence_matrix_learn(configs)
        self.dec_mapping=nn.Linear(self.word_embeddings.shape[1], configs.enc_in)
        self.len_mapping=nn.Linear(self.dec_alllen,self.seq_len)


    # def set_static_freq_indices(self, static_freqs_idx):
    #     self.static_freq_indices = static_freqs_idx

    def set_static_freqs_idx(self, static_freqs_idx,top_period):
        for i in range(self.layer):
            try:
                self.model[i].static_filter.set_static_freqs_idx(static_freqs_idx)
                self.static_freq_indices = top_period
            except:
                pass

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B,T,N=x_enc.size()
        # Normalization from Non-stationary Transformer.
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        x_enc=self.dim_mapping(x_enc)
        all_out=[]
        for i in range(self.top_K_static_freqs):
            period=self.static_freq_indices[i]
            if (self.seq_len)% period != 0:
                length = (((self.seq_len) // period) + 1) * period
                padding = torch.zeros([x_enc.shape[0], (length - self.seq_len), x_enc.shape[2]]).to(x_enc.device)
                out = torch.cat([x_enc, padding], dim=1)
                all_out.append(out)
            else:
                length = (self.seq_len)
                out = x_enc
                all_out.append(out)
        ###multi-scale construction

        seq_enc=self.multi_conv(all_out)
        enc_in = self.layer_norm(self.enc_embedding(x_enc, x_mark_enc))
        word_all=[]
        concat_all = []
        # concat_pca=[]
        word_output=self.word_embeddings

        for i in range (len(self.word_size)):
            word_output=self.multi_word_layer[i](word_output.permute(1,0)).permute(1,0)


            word_all.append(word_output)
            word_expanded=word_output.unsqueeze(0).expand(self.batch_size, -1, -1).permute(0, 2, 1)
            result = torch.cat((seq_enc[i].permute(0, 2, 1), word_expanded), dim=-1).permute(0, 2, 1)
            concat_all.append(result)
            # node_num.append(result.shape[2])

        word_new=torch.cat(word_all,dim=0)
        dec_out = word_new.detach().cpu().numpy()
        emb = StandardScaler().fit_transform(dec_out)
        dec_out = emb.reshape(-1, dec_out.shape[-1])

        emb = PCA(n_components=50, random_state=42).fit_transform(dec_out)
        # tsne = TSNE(n_components=2, perplexity=30, learning_rate=500, n_iter=1000, random_state=42)
        # tsne = TSNE(n_components=2, random_state=42)
        tsne = TSNE(n_components=2, perplexity=10, learning_rate=10, n_iter=3000, init='random', random_state=24,
                    early_exaggeration=50, metric='cosine', n_jobs=-1)
        embeddings_2d = tsne.fit_transform(dec_out)

        # 可视化降维后的嵌入
        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],s=10)

        plt.title('t-SNE Visualization of GPT-2 Word Embeddings (Optimized)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig('./3.png')
        plt.show()
        #### 点边关联矩阵构建
        # incidence_matrix=torch.tensor(self.incidence(seq_enc,word_all)).to(seq_enc.device)
        incidence_matrix,loss_con = self.incidence(seq_enc, word_all)

        """
        ####约束损失设计
        concat_dim=concat_all[0].reshape(-1,768)
        pos_indices_local=build_pos_indices_from_edge_index(incidence_matrix[0], num_nodes_per_sample=concat_all[0].shape[1])
        loss = acl_loss_batched(concat_dim, pos_indices_local, num_nodes_per_sample=concat_all[0].shape[1], tau=0.1)
        print("ACL Loss:", loss.item())
        """
        loss_con/=self.batch_size
        ####超图卷积
        # incidence_matrix= torch.tensor(adj_matrix[i]).to(x.device)
        out_all=[]
        for i in range(len(self.word_size)):
            output = self.hyperatten[i](concat_all[i], incidence_matrix[i])
            out_all.append(output)

        #####loss设计


        ####输出拼接（后面加入prompts）
        # ccc=out_all
        multi_all = torch.cat(out_all, dim=1)
        dec_out = multi_all.detach().cpu().numpy()
        dec_out = dec_out.reshape(-1, dec_out.shape[-1])
        dec_out = dec_out[:5000]
        # tsne = TSNE(n_components=2, perplexity=30, learning_rate=500, n_iter=1000, random_state=42)
        # tsne = TSNE(n_components=2, random_state=42)
        pca = PCA(n_components=50, random_state=42)
        # tsne = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=1000, init='random', random_state=42, early_exaggeration=50, n_jobs=-1)
        tsne = TSNE(n_components=2, perplexity=10, learning_rate=10, n_iter=3000, init='random', random_state=24,
                    early_exaggeration=50, metric='cosine', n_jobs=-1)
        dec_out=pca.fit_transform(dec_out)
        embeddings_2d = tsne.fit_transform(dec_out)

        # n_clusters = 8
        # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # labels = kmeans.fit_predict(embeddings_2d)
        #
        # plt.figure(figsize=(8, 6))
        # scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
        #                       c=labels, cmap='tab10', s=10, alpha=0.8)
        # plt.title("t-SNE + KMeans Clustering", fontsize=14)
        # plt.axis('off')
        # plt.savefig('./tsne_embeddings1112.png')
        # plt.show()

        # 可视化降维后的嵌入
        plt.figure(figsize=(10, 10))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

        plt.title('t-SNE Visualization of GPT-2 Word Embeddings (Optimized)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        plt.savefig('./3.png')
        plt.show()


        ####LLMs输出
        dec_out = self.llm_model(inputs_embeds=multi_all).last_hidden_state
        ####维度转换
        dec_out=self.dec_mapping(dec_out)
        dec_out=self.len_mapping(dec_out.permute(0,2,1)).permute(0,2,1)




        """
        ###original(原来的代码)
        for i in range(self.layer):
            enc_out = self.model[i](enc_in)
            enc_out = self.layer_norm(enc_out)
        #out (B, N, d_model)
        dec_in = enc_out + enc_in
        dec_out = self.projection(dec_in)
        dec_out = dec_out.transpose(2,1)[:, :, :self.c_out]
        """

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len, 1))
            dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :],loss_con




class incidence_matrix_learn(nn.Module):
    def __init__(self,configs):
        super(incidence_matrix_learn, self).__init__()
        self.seq_len = configs.seq_len
        self.window_size=configs.window_size
        self.inner_size=configs.inner_size
        self.dim=configs.d_model
        self.hyper_num=configs.hyper_num
        self.alpha=3
        self.k=configs.k
        self.embedhy=nn.ModuleList()
        self.embednod=nn.ModuleList()
        self.linhy=nn.ModuleList()
        self.linnod=nn.ModuleList()
        self.word_size = configs.word_size

        self.top_period = configs.top_period
        self.top_K_static_freqs = configs.top_K_static_freqs

        all_out=[]

        for i in range(len(self.word_size)):
            period=self.top_period[i]
            output=torch.tensor((self.seq_len // period if self.seq_len % period == 0 else math.ceil(self.seq_len / period)) + self.word_size[i])

            all_out.append(output)

        self.nodnum=all_out
            # getattr(configs, 'top_period', [1])
        # self.extend=nn.Linear
        for i in range(len(self.hyper_num)):
            # self.embednod.append(nn.Linear(self.nodnum[i], self.dim))
            self.embedhy.append(nn.Embedding(self.hyper_num[i],self.dim))
            self.linhy.append(nn.Linear(self.dim,self.dim))
            self.linnod.append(nn.Linear(self.dim,self.dim))
            self.embednod.append(nn.Embedding(self.nodnum[i],self.dim))

            # if i==0:
            #     self.embednod.append(nn.Embedding(self.seq_len,self.dim))
            # else:
            #     product=math.prod(self.window_size[:i])
            #     layer_size=math.floor(self.seq_len/product)
            #     self.embednod.append(nn.Embedding(int(layer_size),self.dim))

        self.dropout = nn.Dropout(p=0.1)

    def forward(self,x,word):
        node_num = []
        # node_num.append(self.seq_len)

        # for i in range(len(self.window_size)):
        #     layer_size = math.floor(node_num[i] / self.window_size[i])
        #     node_num.append(layer_size)
        hyperedge_all=[]
        B=x[1].size(0)
        concat_all=[]
        ####先将时间序列和文本表示拼接
        for i in range(len(word)):
            word_expanded=word[i].unsqueeze(0).expand(B,-1,-1).permute(0,2,1)
            result=torch.cat((x[i].permute(0,2,1),word_expanded),dim=-1)
            concat_all.append(result)
            node_num.append(result.shape[2])
            # print("okkk")

        loss_all=0
        for i in range(len(self.hyper_num)):
            hypidxc=torch.arange(self.hyper_num[i]).to(x[0].device)
            nodeidx=torch.arange(node_num[i]).to(x[0].device)
            hyperen=self.embedhy[i](hypidxc)
            nodeec=self.embednod[i](nodeidx)

            a = torch.mm(nodeec, hyperen.transpose(1, 0))
            adj=F.softmax(F.relu(self.alpha*a))
            mask = torch.zeros(nodeec.size(0), hyperen.size(0)).to(x[0].device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(min(adj.size(1),self.k), 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask
            adj = torch.where(adj > 0.5, torch.tensor(1).to(x[0].device), torch.tensor(0).to(x[0].device))
            adj = adj[:, (adj != 0).any(dim=0)]
            matrix_array = torch.tensor(adj, dtype=torch.int)
            result_list = [list(torch.nonzero(matrix_array[:, col]).flatten().tolist()) for col in
                           range(matrix_array.shape[1])]

            node_list = torch.cat([torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]).tolist()
            count_list = list(torch.sum(adj, dim=0).tolist())
            hperedge_list = torch.cat([torch.full((count,), idx) for idx, count in enumerate(count_list, start=0)]).tolist()
            hypergraph=np.vstack((node_list,hperedge_list))

            hypergraph=torch.tensor(hypergraph).to(x[0].device)

            ####约束损失设计
            concat_dim = concat_all[i].permute(0,2,1)
            concat_dim=concat_dim.reshape(-1, concat_dim.size(-1))
            pos_indices_local = build_pos_indices_from_edge_index(hypergraph,
                                                                  num_nodes_per_sample=concat_all[i].shape[1])
            loss = acl_loss_batched(concat_dim, pos_indices_local, num_nodes_per_sample=concat_all[i].shape[1], tau=0.1)
            # print("ACL Loss:", loss.item())
            loss_all+=loss
            hyperedge_all.append(hypergraph)

        # ccc=hyperedge_all
        # print(ccc)

        return hyperedge_all,loss_all


class HyperAtten(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.1,
                 bias=False):
        super(HyperAtten, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention


        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels))

            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))

        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:

            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def __forward__(self,
                    x,
                    hyperedge_index,
                    alpha=None):

        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        # --------------------------------------------------------
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)

        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out=alpha.unsqueeze(-1)*out
        return out
    def forward(self, x, hyperedge_index):
        x = torch.matmul(x, self.weight)
        # hyperedge_index=torch.tensor(hyperedge_index).to(x.device)
        x1=x.transpose(0,1)
        x_i = torch.index_select(x1, dim=0, index=torch.tensor(hyperedge_index[0], device=x.device))
        edge_sums = {}
        # print("max edge_id:", max(hyperedge_index[0]))
        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            if edge_id not in edge_sums:
                edge_id = edge_id.item()
                node_id = node_id.item()
                edge_sums[edge_id] = x1[node_id, :, :]
            else:
                edge_sums[edge_id] += x1[node_id, :, :]
        result_list = torch.stack([value for value in edge_sums.values()], dim=0)
        # x_j = torch.index_select(result_list, dim=0, index=torch.tensor(hyperedge_index[0], device=x.device))
        x_j = torch.index_select(result_list, dim=0, index=hyperedge_index[1])
        # loss_hyper = 0
        # for k in range(len(edge_sums)):
        #     for m in range(len(edge_sums)):
        #         inner_product = torch.sum(edge_sums[k] * edge_sums[m], dim=1, keepdim=True)
        #         norm_q_i = torch.norm(edge_sums[k], dim=1, keepdim=True)
        #         norm_q_j = torch.norm(edge_sums[m], dim=1, keepdim=True)
        #         alpha = inner_product / (norm_q_i * norm_q_j)
        #         distan = torch.norm(edge_sums[k] - edge_sums[m],dim=1, keepdim=True)
        #         loss_item = alpha * distan + (1 - alpha) * (torch.clamp(torch.tensor(4.2) - distan, min=0.0))
        #         loss_hyper += torch.abs(torch.mean(loss_item))


        # loss_hyper = loss_hyper / ((len(edge_sums) + 1)**2)
        # print(x_i.shape)  # 例如 [N, F]
        # print(x_j.shape)
        # print(self.att.shape)
        # cat = torch.cat([x_i, x_j], dim=-1)
        # print("cat.shape:", cat.shape)
        #
        # if torch.isnan(cat).any() or torch.isinf(cat).any():
        #     print("❌ cat contains NaN or Inf!")
        #
        # att_applied = cat * self.att
        # if torch.isnan(att_applied).any() or torch.isinf(att_applied).any():
        #     print("❌ att_applied contains NaN or Inf!")
        #
        # alpha = att_applied.sum(dim=-1)
        # if torch.isnan(alpha).any() or torch.isinf(alpha).any():
        #     print("❌ alpha contains NaN or Inf!")
        #
        # print("✅ All values look good")

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)####alpha=[594,32] hyperedge_index[0]=594 x.size(0)=[32]
        # index = torch.tensor(hyperedge_index[0], dtype=torch.long, device=x1.device)
        # index = torch.tensor(hyperedge_index[0], device=x1.device)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x1.size(0))
        # print(torch.isnan(alpha).any())
        # print(alpha.device)
        # alpha=torch.tensor(alpha).to(x1.device)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        D = degree(hyperedge_index[0], x1.size(0), x1.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x1.dtype)
        B[B == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x1, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        out=out.transpose(0, 1)
        # constrain_loss = x_i - x_j
        # constrain_lossfin1=torch.mean(constrain_loss)
        # constrain_losstotal = abs(constrain_lossfin1) + loss_hyper
        return out
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




def build_pos_indices_from_edge_index(edge_index, num_nodes_per_sample):
    """
    构建局部正样本索引，形如 [[2,3], [1,3], [1,2], ...]
    """
    node_idx, hedge_idx = edge_index
    hedge_to_nodes = defaultdict(list)

    for n, h in zip(node_idx.tolist(), hedge_idx.tolist()):
        hedge_to_nodes[h].append(n)

    pos_dict = defaultdict(set)
    for nodes in hedge_to_nodes.values():
        for i in nodes:
            for j in nodes:
                if i != j:
                    pos_dict[i].add(j)

    pos_indices = []
    for i in range(num_nodes_per_sample):
        pos_indices.append(list(pos_dict[i]))

    return pos_indices


def acl_loss_batched(node_embeddings, pos_indices_local, num_nodes_per_sample, tau=0.1):
    """
    node_embeddings: [B*N, D]
    pos_indices_local: List[List[int]], 每个样本共享的正样本集合（0~N-1 局部编号）
    """
    B = node_embeddings.shape[0] // num_nodes_per_sample
    D = node_embeddings.shape[1]
    loss = 0.0
    valid_count = 0

    for b in range(B):
        # 取出该样本的节点特征 [N, D]
        start = b * num_nodes_per_sample
        end = (b + 1) * num_nodes_per_sample
        n = node_embeddings[start:end]

        # 归一化后计算 [N, N] 相似度矩阵
        sim = torch.matmul(F.normalize(n, dim=1), F.normalize(n, dim=1).T) / tau

        for i in range(num_nodes_per_sample):
            pos_i = pos_indices_local[i]
            if len(pos_i) == 0:
                continue

            denom = torch.logsumexp(sim[i], dim=0)
            pos_sims = sim[i, pos_i]
            loss_i = - (pos_sims - denom).mean()

            loss += loss_i
            valid_count += 1

    return loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=node_embeddings.device)
