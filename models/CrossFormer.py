import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder, FeedForward, FixedFrequencyEncoder
from utils.new_neighbor_sampler import NeighborSampler
import utils.globals as globals


class CrossFormer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu', hops: int = 2):
        """
        DyGFormer model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param dim_encode: int, dimension of time features or freq features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(CrossFormer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1] if hops == 1 else self.node_raw_features.shape[1] + 2
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device
        self.hops = hops

        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim)
        self.frequency_encoder = FixedFrequencyEncoder(channel_embedding_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        
        # self.neighbor_co_occurrence_encode_layer = nn.Sequential(
        #         nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
        #         nn.ReLU(),
        #         nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim))
        
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': FeedForward(dims=self.patch_size * (self.neighbor_co_occurrence_feat_dim+self.max_input_sequence_length), out_dims=self.channel_embedding_dim, dropout=0., use_single_layer=True)
            # nn.Linear(in_features=self.patch_size * (self.neighbor_co_occurrence_feat_dim+self.max_input_sequence_length), out_features=self.channel_embedding_dim, bias=True)
        })
        
        self.num_channels = 4

        self.transformers = nn.ModuleList([
            nn.ModuleList([
                TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads, dropout=self.dropout)
                for _ in range(2)
            ])
            # TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.ModuleList([
            nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True),
            nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)
        ])
        

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, no_time: bool=False):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        if not no_time:
            globals.timer.start_neighbor_sample()
        
        if(self.hops == 2):
            self.neighbor_sampler.set_fanouts([self.max_input_sequence_length//4, 3])
        else:
            self.neighbor_sampler.set_fanouts([self.max_input_sequence_length,])
        src_node_ids_th, dst_node_ids_th, node_interact_times_th = torch.from_numpy(src_node_ids), torch.from_numpy(dst_node_ids), torch.from_numpy(node_interact_times)
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        self.neighbor_sampler.neighbor_sample_from_nodes(src_node_ids_th, node_interact_times_th)
        # Shape: (batch_size, max_input_sequence_length)
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list, src_srcindex_list = \
            self.neighbor_sampler.get_ret()
            # self.neighbor_sampler.get_multi_hop_neighbors(num_hops=2, node_ids=src_node_ids, node_interact_times=node_interact_times, num_neighbors=[self.max_input_sequence_length//4, 3])
            # self.neighbor_sampler.sample_recent_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times, num_neighbors=self.max_input_sequence_length)
            # self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)
        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        self.neighbor_sampler.neighbor_sample_from_nodes(dst_node_ids_th, node_interact_times_th)
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list, dst_srcindex_list = \
            self.neighbor_sampler.get_ret()
            # self.neighbor_sampler.get_multi_hop_neighbors(num_hops=2, node_ids=dst_node_ids, node_interact_times=node_interact_times, num_neighbors=[self.max_input_sequence_length//4, 3])
            # self.neighbor_sampler.sample_recent_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times, num_neighbors=self.max_input_sequence_length)
            # self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)
        # print((src_nodes_neighbor_ids_list!=0).sum(axis=1))
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = src_nodes_neighbor_ids_list.numpy(), src_nodes_edge_ids_list.numpy(), src_nodes_neighbor_times_list.numpy()
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = dst_nodes_neighbor_ids_list.numpy(), dst_nodes_edge_ids_list.numpy(), dst_nodes_neighbor_times_list.numpy()

        if not no_time:
            globals.timer.end_neighbor_sample()

        if not no_time:
            globals.timer.start_construct_patchs()
        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)
        if not no_time:
            globals.timer.end_construct_patchs()

        if not no_time:
            globals.timer.start_encodeCo()
        from utils.cpp.src.cpp_cores import count_nodes_cooccurrence
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = count_nodes_cooccurrence(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids, 10)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = torch.from_numpy(src_padded_nodes_appearances).float().to(self.device), torch.from_numpy(dst_padded_nodes_appearances).float().to(self.device)
        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features = self.frequency_encoder(src_padded_nodes_appearances).sum(dim=2)
        # Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        dst_padded_nodes_neighbor_co_occurrence_features = self.frequency_encoder(dst_padded_nodes_appearances).sum(dim=2)
        # add identity encoding for the same nodes
        src_padded_nodes_neighbor_ids_th = torch.from_numpy(src_padded_nodes_neighbor_ids).to(self.device)
        src_neigh_mask = src_padded_nodes_neighbor_ids_th.unsqueeze(1) == src_padded_nodes_neighbor_ids_th.unsqueeze(2)
        src_iden_encode = src_neigh_mask.float()
        src_padded_nodes_neighbor_co_occurrence_features = torch.cat([src_padded_nodes_neighbor_co_occurrence_features, src_iden_encode], dim=2)
        dst_padded_nodes_neighbor_ids_th = torch.from_numpy(dst_padded_nodes_neighbor_ids).to(self.device)
        dst_neigh_mask = dst_padded_nodes_neighbor_ids_th.unsqueeze(1) == dst_padded_nodes_neighbor_ids_th.unsqueeze(2)
        dst_iden_encode = dst_neigh_mask.float()
        dst_padded_nodes_neighbor_co_occurrence_features = torch.cat([dst_padded_nodes_neighbor_co_occurrence_features, dst_iden_encode], dim=2)
        # src_padded_nodes_neighbor_co_occurrence_features = torch.zeros(src_padded_nodes_neighbor_ids.shape[0], src_padded_nodes_neighbor_ids.shape[1], self.neighbor_co_occurrence_feat_dim + self.max_input_sequence_length, device=self.device)
        # dst_padded_nodes_neighbor_co_occurrence_features = torch.zeros(dst_padded_nodes_neighbor_ids.shape[0], dst_padded_nodes_neighbor_ids.shape[1], self.neighbor_co_occurrence_feat_dim + self.max_input_sequence_length, device=self.device)
        if not no_time:
            globals.timer.end_encodeCo()
        
        if not no_time:
            globals.timer.start_load_feature()
        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)
        if not no_time:
            globals.timer.end_load_feature()
            
        # add the source index to the features if hops == 2
        if(self.hops == 2):
            src_srcindex_list = src_srcindex_list.view(src_padded_nodes_neighbor_node_raw_features.shape[0], 
                                                    src_padded_nodes_neighbor_node_raw_features.shape[1], -1).float().to(self.device)
            src_padded_nodes_neighbor_node_raw_features = torch.cat([src_padded_nodes_neighbor_node_raw_features, 
                                                                    src_srcindex_list,
                                                                    (src_srcindex_list==0).float()], 
                                                                    dim=2)
            dst_srcindex_list = dst_srcindex_list.view(dst_padded_nodes_neighbor_node_raw_features.shape[0],
                                                    dst_padded_nodes_neighbor_node_raw_features.shape[1], -1).float().to(self.device)
            dst_padded_nodes_neighbor_node_raw_features = torch.cat([dst_padded_nodes_neighbor_node_raw_features,
                                                                    dst_srcindex_list,
                                                                    (dst_srcindex_list==0).float()],
                                                                    dim=2)
        
        if not no_time:
            globals.timer.start_construct_patchs()
        # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
        # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)
        if not no_time:
            globals.timer.end_construct_patchs()

        # align the patch encoding dimension
        # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        if src_patches_nodes_neighbor_co_occurrence_features is not None:
            src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        if dst_patches_nodes_neighbor_co_occurrence_features is not None:
            dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        # patches_nodes_neighbor_node_raw_features = torch.cat([src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        # patches_nodes_edge_raw_features = torch.cat([src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        # patches_nodes_neighbor_time_features = torch.cat([src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        # patches_nodes_neighbor_co_occurrence_features = torch.cat([src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features], dim=1)

        src_patches_data = [src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features,
                        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features]
        dst_patches_data = [dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features,
                            dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features]
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        # patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features]
        # patches_data = torch.stack(patches_data, dim=2)
        src_patches_data = torch.stack(src_patches_data, dim=2)
        dst_patches_data = torch.stack(dst_patches_data, dim=2)
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        # patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)
        src_patches_data = src_patches_data.reshape(batch_size, src_num_patches, self.num_channels * self.channel_embedding_dim)
        dst_patches_data = dst_patches_data.reshape(batch_size, dst_num_patches, self.num_channels * self.channel_embedding_dim)

        if not no_time:
            globals.timer.start_transform()
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        for transformer in self.transformers:
            src_patches_data = transformer[0](dst_patches_data, src_patches_data)
            dst_patches_data = transformer[1](src_patches_data, dst_patches_data)
            # patches_data = transformer(patches_data, patches_data, patches_data)
        if not no_time:
            globals.timer.end_transform()
            
        # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        # src_patches_data = patches_data[:, : src_num_patches, :]
        # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        # dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        src_patches_data = torch.mean(src_patches_data, dim=1)
        # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer[0](src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer[1](dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])-1

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_input_sequence_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_input_sequence_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_input_sequence_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                node_len = min(len(nodes_neighbor_ids_list[idx])-1, max_input_sequence_length-1)
                padded_nodes_neighbor_ids[idx, 1: node_len + 1] = nodes_neighbor_ids_list[idx][-node_len:]
                padded_nodes_edge_ids[idx, 1: node_len + 1] = nodes_edge_ids_list[idx][-node_len:]
                padded_nodes_neighbor_times[idx, 1: node_len + 1] = nodes_neighbor_times_list[idx][-node_len:]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features = [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            if padded_nodes_neighbor_co_occurrence_features is not None:
                patches_nodes_neighbor_co_occurrence_features.append(padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)
        if padded_nodes_neighbor_co_occurrence_features is not None:
            patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * (self.neighbor_co_occurrence_feat_dim+self.max_input_sequence_length))
        else:
            patches_nodes_neighbor_co_occurrence_features = None
        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class TransformerEncoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1, num_channels: int = 4):
        """
        TransformerEncoder constructor.

        Args:
            attention_dim (int): The dimension of the attention vectors.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = nn.ModuleList([
            MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout, batch_first=True),
            MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout, batch_first=True),
            MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        ])            
            
        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList([
            FeedForward(dims=attention_dim, out_dims=attention_dim, dropout=dropout, expansion_factor=2),
            FeedForward(dims=attention_dim, out_dims=attention_dim, dropout=dropout, expansion_factor=2),
            nn.Linear(in_features=attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, q: torch.Tensor, kv: torch.Tensor, pad_mask: torch.Tensor = None):
        """
        Forward pass of the TransformerEncoder.

        Args:
            q (torch.Tensor): The queries. Shape: (batch_size, seq_len, attention_dim).
            kv (torch.Tensor): The keys and values. Shape: (batch_size, seq_len, attention_dim).
            pad_mask (torch.Tensor, optional): The padding mask. Defaults to None. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: The output of the encoder. Shape: (batch_size, seq_len, attention_dim).
        """
        # Pass through the multi-head attention layer
        decoder = self.multi_head_attention[0](query=q, key=q, value=q, key_padding_mask=pad_mask)[0]  # Shape: (batch_size, seq_len, attention_dim)
        # Add and normalize
        decoder = self.norm_layers[0](self.dropout(decoder) + q)
        
        # Pass through the multi-head attention layer
        encoder = self.multi_head_attention[1](query=kv, key=kv, value=kv, key_padding_mask=pad_mask)[0]  # Shape: (batch_size, seq_len, attention_dim)
        # Add & normalize
        encoder = self.norm_layers[1](self.dropout(encoder) + kv)
        # FFN & ADD & normalize
        encoder = self.norm_layers[2](self.dropout(self.linear_layers[0](encoder)) + encoder)
        
        # Cross Attention
        hidden_states = self.multi_head_attention[2](query=decoder, key=encoder, value=encoder, key_padding_mask=pad_mask)[0]  # Shape: (batch_size, seq_len, attention_dim)
        # Add & normalize
        hidden_states = self.norm_layers[3](self.dropout(hidden_states) + q)
        # FNN & ADD & normalize
        hidden_states = self.norm_layers[4](self.dropout(self.linear_layers[1](hidden_states)) + hidden_states)
        
        # Pass through the feed-forward network
        outputs = F.gelu(self.dropout(self.linear_layers[2](hidden_states)))  # Shape: (batch_size, seq_len, attention_dim)
        
        return outputs