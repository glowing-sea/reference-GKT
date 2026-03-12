from hashlib import new
import math
from matplotlib.patheffects import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


# Multi-Layer Perceptron(MLP) layer
class MLP(nn.Module):
    """Two-layer fully-connected ReLU net with batch norm."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        # the paper said they added Batch Normalization for the output of MLPs, as shown in Section 4.2
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 or 0 will cause BatchNorm error, so return the input directly
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:  # len(input_size()) == 2
            return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)  # pay attention to add training=self.training
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    """
    Erase & Add Gate module
    NOTE: this erase & add gate is a bit different from that in DKVMN.
    For more information about Erase & Add gate, please refer to the paper "Dynamic Key-Value Memory Networks for Knowledge Tracing"
    The paper can be found in https://arxiv.org/abs/1611.08108
    """

    def __init__(self, feature_dim, concept_num, bias=True):
        super(EraseAddGate, self).__init__()
        # weight
        self.weight = nn.Parameter(torch.rand(concept_num))
        self.reset_parameters()
        # erase gate
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        # add gate
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        r"""
        Params:
            x: input feature matrix
        Shape:
            x: [batch_size, concept_num, feature_dim]
            res: [batch_size, concept_num, feature_dim]
        Return:
            res: returned feature matrix with old information erased and new information added
        The GKT paper didn't provide detailed explanation about this erase-add gate. As the erase-add gate in the GKT only has one input parameter,
        this gate is different with that of the DKVMN. We used the input matrix to build the erase and add gates, rather than $\mathbf{v}_{t}$ vector in the DKVMN.
        """
        erase_gate = torch.sigmoid(self.erase(x))  # [batch_size, concept_num, feature_dim]
        # self.weight.unsqueeze(dim=1) shape: [concept_num, 1]
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))  # [batch_size, concept_num, feature_dim]
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res





# READ
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    NOTE: Stole and modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    """

    def __init__(self, temperature, # In standard scaled dot-product attention, temperature = sqrt(d_k) to prevent large dot-product values
                 # Without it, softmax becomes extremely peaky, leading to small gradients
                 # Var(Q K / sqrt(d_k)) = 1
                 attn_dropout=0.): # a [n_head, mask_num, concept_num] matrix with all 1 everywhere except mask[k][i][masked_qt[i]] = 0
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(self, q, k, mask=None):
        r"""
        Parameters:
            q: multi-head query matrix
            k: multi-head key matrix
            mask: mask matrix
        Shape:
            q: [n_head, mask_num, embedding_dim]
            k: [n_head, concept_num, embedding_dim] -> k.T: [n_head, embedding_dim, concept_num]
            so atten(q, k): [n_head, mask_number, concept_num]
        Return: attention score of all queries
        """
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))  # [n_head, mask_number, concept_num]

        # To ensure no self loop
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # pay attention to add training=self.training!

        # Most attention implementations apply:
        # softmax(attn, dim = -1), i.e., Each query’s attention over all concepts sums to 1
        # The softmax is over the heads (edge types)
        # Drop out is applied only during training
        attn = F.dropout(F.softmax(attn, dim=0), self.dropout, training=self.training)  # pay attention that dim=-1 is not as good as dim=0!
        return attn



# READ
# The encoder takes concept embeddings x0,x1,…,xC−1 and produces logits for every possible directed concept pair (i→j)
class MLPEncoder(nn.Module):
    # Take two concept embeddings (sender → receiver)
    # and produce a vector of logits telling
    # what edge type the edge belongs to.
    """
    MLP encoder module.
    NOTE: Stole and modify the code from https://github.com/ethanfetaya/NRI/blob/master/modules.py
    """
    # hidden_dim: args.vae_encoder_dim
    def __init__(self, input_dim, hidden_dim, output_dim, factor=True, dropout=0., bias=True):
        super(MLPEncoder, self).__init__()
        self.factor = factor # whether to use full factor graph encoder or a simpler encoder
        self.mlp = MLP(input_dim * 2, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.mlp2 = MLP(hidden_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        if self.factor:
            self.mlp3 = MLP(hidden_dim * 3, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        else:
            self.mlp3 = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.init_weights()



    # Default PyTorch (nn.Linear)   Uniform     Depends on fan_in
    # Xavier Normal                 Normal      Depends on fan_in + fan_out
    # He/Kaiming Normal             Normal      fan_in
    def init_weights(self):
        # This walks through every submodule recursively, so:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x, sp_send, sp_rec):
        # NOTE: Assumes that we have the same graph across all samples.


        # sp_rec: [edge_num, concept_num]
        # x:      [concept_num, embedding_dim]
        # -----------------------------------
        # receivers: [edge_num, embedding_dim]

        receivers = torch.matmul(sp_rec, x) # for each edge, the embedding of its receiver node
        senders = torch.matmul(sp_send, x) # for each edge, the embedding of its sender node
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, sp_send_t, sp_rec_t):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(sp_rec_t, x)
        return incoming

    # Neural Relational Inference
    def forward(self, inputs, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            inputs: input concept embedding matrix
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
        Shape:
            inputs: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]       # 1 in the 1st row and 2nd column, it means the sender of edge 0 is node 1
            sp_rec: [edge_num, concept_num]        # edg_num = 2 * concept_num * (concept_num - 1)
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            output: [edge_num, edge_type_num]
        """
        # inputs in [concept_num, embedding_dim]
        x = self.node2edge(inputs, sp_send, sp_rec)  # [edge_num, 2 * embedding_dim]
        x = self.mlp(x)  # edge features, shape [E, H]
        x_skip = x # to create skip connection later

        if self.factor:

            # Concatenation allows the final representation to:
            # retain raw edge information (from embeddings)
            # include message-passed relational information (via nodes)
            # learn to weight/combine them through the following MLP

            # This introduces global relational reasoning:
            # Edges influence node states
            # Node states influence other edges
            # This allows detection of higher-order relational structures.

            # For each node j,
            # Aggregate all incoming edges i → j
            # Result is node feature: [C, H]
            x = self.edge2node(x, sp_send_t, sp_rec_t)  # [concept_num, hidden_num]

            x = self.mlp2(x)  # [concept_num, hidden_num] # mix information between edges via nodes
            x = self.node2edge(x, sp_send, sp_rec)  # [edge_num, 2 * hidden_num]
            x = torch.cat((x, x_skip), dim=1)  # Skip connection  shape: [edge_num, 3 * hidden_num]
            x = self.mlp3(x)  # [edge_num, hidden_num]

        else: # Simple edge classifier
            # x has shape [E, H] (new edge features)
            # x_skip has shape [E, H] (original edge features from earlier)
            
            # ZERO message passing between edge. Data flows within a single edge only.

            x = self.mlp2(x)  # [edge_num, hidden_num]
            x = torch.cat((x, x_skip), dim=1)  # Skip connection  shape: [edge_num, 2 * hidden_num]
            x = self.mlp3(x)  # [edge_num, hidden_num]

        output = self.fc_out(x)  # [edge_num, output_dim] = [edge_num, edge_type_num]
        return output


# Concept embeddings (C x D)
#         │
# node2edge
#         ▼
# Edge features (E x 2D)
#         │
#       MLP1
#         ▼
# Edge features h_e (E x H)
#         │
#    (skip saved)
#         │
# edge2node (aggregate incoming messages)
#         ▼
# Node features (C x H)
#         │
#       MLP2
#         ▼
# Node features (C x H)
#         │
# node2edge
#         ▼
# Edge features (E x 2H)
#         │ concat with skip (E x H)
#         ▼
# Concatenated edge feature (E x 3H)
#         │
#       MLP3
#         ▼
# Edge features (E x H)
#         │
#    Linear → output logits
#         ▼
# Final logits (E x K)



# DONE
class MLPDecoder(nn.Module):
    """
    MLP decoder module.
    NOTE: Stole and modify the code from https://github.com/ethanfetaya/NRI/blob/master/modules.py
    """

    def __init__(self, input_dim, msg_hidden_dim, msg_output_dim, hidden_dim, edge_type_num, dropout=0., bias=True):
        super(MLPDecoder, self).__init__()
        self.msg_out_dim = msg_output_dim
        self.edge_type_num = edge_type_num
        self.dropout = dropout

        # For each edge type k, create two MLPs layers
        # Layer 1 maps [xi || xj] in R^{2D} to R^{msg_hidden_dim}
        # Layer 2 maps hidden → output message dim (msg_output_dim)
        self.msg_fc1 = nn.ModuleList([nn.Linear(2 * input_dim, msg_hidden_dim, bias=bias) for _ in range(edge_type_num)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(msg_hidden_dim, msg_output_dim, bias=bias) for _ in range(edge_type_num)])

        # These are edge-type-specific message functions.
        # output x_hat in R^{D}
        self.out_fc1 = nn.Linear(msg_output_dim, hidden_dim, bias=bias)
        self.out_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_fc3 = nn.Linear(hidden_dim, input_dim, bias=bias)

    def node2edge(self, x, sp_send, sp_rec):
        receivers = torch.matmul(sp_rec, x)  # [edge_num, embedding_dim]
        senders = torch.matmul(sp_send, x)  # [edge_num, embedding_dim]
        edges = torch.cat([senders, receivers], dim=-1)  # [edge_num, 2 * embedding_dim]
        return edges

    def edge2node(self, x, sp_send_t, sp_rec_t):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(sp_rec_t, x)
        return incoming

    def forward(self, inputs, rel_type, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            inputs: input concept embedding matrix
            rel_type: inferred edge weights for all edge types from MLPEncoder
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)

            inputs	    [C, D]	Concept embeddings (the original data we want to reconstruct)
            rel_type	[E, K]	Edge type assignments for each edge (sampled by Gumbel-Softmax)
            sp_send	    [E, C]	1-hot: sender node index for each edge
            sp_rec	    [E, C]	1-hot: receiver node index for each edge
            sp_send_t	[C, E]	transpose of sp_send
            sp_rec_t	[C, E]	transpose of sp_rec

        Shape:
            inputs: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]
            sp_rec: [edge_num, concept_num]
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            output: [edge_num, edge_type_num]
        """
        # NOTE: Assumes that we have the same graph across all samples.
        # Node2edge

        # Build pre-messages for each directed edge
        pre_msg = self.node2edge(inputs, sp_send, sp_rec)  # [edge_num, 2 * embedding_dim] = [E, 2D]

        # Initialize all messages to zero
        all_msgs = Variable(torch.zeros(pre_msg.size(0), self.msg_out_dim, device=inputs.device))  # [edge_num, msg_out_dim]

        # m_ij = sum from k=1 to K of (msg_ij^k * z_ij^k)
        for i in range(self.edge_type_num):
            msg = F.relu(self.msg_fc1[i](pre_msg)) # [E, msg_hidden]
            msg = F.dropout(msg, self.dropout, training=self.training)
            msg = F.relu(self.msg_fc2[i](msg)) # [E, msg_out_dim]
            msg = msg * rel_type[:, i:i + 1] # [E, msg_out_dim] * [E, 1] and broadcast to [E, msg_out_dim]
            all_msgs += msg # [E, msg_out_dim]

        # Aggregate all msgs to receiver
        agg_msgs = self.edge2node(all_msgs, sp_send_t, sp_rec_t)  # [concept_num, msg_out_dim]
        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(agg_msgs)), self.dropout, training=self.training)  # [concept_num, hidden_dim]
        pred = F.dropout(F.relu(self.out_fc2(pred)), self.dropout, training=self.training)  # [concept_num, hidden_dim]
        pred = self.out_fc3(pred)  # [concept_num, embedding_dim]
        return pred # [C, D]
    


# x in (C × D) ---- x is the concept embedding matrix
#    │
# node2edge
#    ▼
# pre_msg (E × 2D)
#    │ per-type MLP
#    ├──> msg_k  (E × M)  ── weight by z_k ──┐
#    ├──> msg_k  (E × M)  ── weight by z_k ──┤ sum → all_msgs
#    └──> msg_k  ...                        ┘
#    ▼
# all_msgs (E × M)
#    │ edge2node
#    ▼
# agg_msgs (C × M)
#    │ MLP
#    ▼
# pred (C × D)
