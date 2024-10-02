import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder, FeedForward, FixedFrequencyEncoder
from utils.new_neighbor_sampler import NeighborSampler
import utils.globals as globals


class QSFormer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, cross_edge_neighbor_feat_dim: int, patch_size: int = 1, 
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, max_input_sequence_length: int = 512, 
                 num_high_order_neighbors: int = 3, device: str = 'cpu', hops: int = 2, no_id_encode = False, dim_expansion_factor: int = 4):
        """
        QSFormer model.
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
        super(QSFormer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1] if hops == 1 or no_id_encode else self.node_raw_features.shape[1] + 2
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.num_high_order_neighbors = num_high_order_neighbors
        self.device = device
        self.hops = hops
        self.no_id_encode = no_id_encode
        self.dim_expansion_factor = dim_expansion_factor
        
        self.cross_edge_neighbor_feat_dim = cross_edge_neighbor_feat_dim

        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim)
        self.frequency_encoder = FixedFrequencyEncoder(self.cross_edge_neighbor_feat_dim)
        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.cross_edge_neighbor_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.cross_edge_neighbor_feat_dim, out_features=self.cross_edge_neighbor_feat_dim))
        
        if self.hops==2:
            self.aggregate_layer = FeedForward(dims=self.cross_edge_neighbor_feat_dim, out_dims=self.cross_edge_neighbor_feat_dim, dropout=dropout, use_single_layer=True)
            
        self.compress_layer = FeedForward(dims=self.max_input_sequence_length//(self.num_high_order_neighbors+1), out_dims=self.cross_edge_neighbor_feat_dim, dropout=dropout, use_single_layer=True)
        
        self.projection_layer = nn.ModuleDict({
            'node': FeedForward(dims=self.patch_size * self.node_feat_dim, out_dims=self.channel_embedding_dim, dropout=dropout, use_single_layer=True),
            'edge': FeedForward(dims=self.patch_size * self.edge_feat_dim, out_dims=self.channel_embedding_dim, dropout=dropout, use_single_layer=True),
            'time': FeedForward(dims=self.patch_size * self.time_feat_dim, out_dims=self.channel_embedding_dim, dropout=dropout, use_single_layer=True),
            'neighbor_co_occurrence': FeedForward(
                dims= self.patch_size * self.cross_edge_neighbor_feat_dim if self.no_id_encode else self.patch_size * (self.cross_edge_neighbor_feat_dim+self.max_input_sequence_length//(self.num_high_order_neighbors+1)), 
                out_dims=self.channel_embedding_dim, 
                dropout=dropout, use_single_layer=True)
        })
        
        self.num_channels = 4

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads, dropout=self.dropout, dim_expansion_factor=self.dim_expansion_factor)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.ModuleList([
            nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_raw_features.shape[1], bias=True),
            nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_raw_features.shape[1], bias=True)
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
            assert self.max_input_sequence_length % (self.num_high_order_neighbors+1) == 0, "max_input_sequence_length should be divisible by num_high_order_neighbors+1"
            self.neighbor_sampler.set_fanouts([self.max_input_sequence_length//(self.num_high_order_neighbors+1), self.num_high_order_neighbors])
        else:
            self.neighbor_sampler.set_fanouts([self.max_input_sequence_length,])
        max_input_sequence_length = self.max_input_sequence_length//(1+self.num_high_order_neighbors)
        src_node_ids_th, dst_node_ids_th, node_interact_times_th = torch.from_numpy(src_node_ids), torch.from_numpy(dst_node_ids), torch.from_numpy(node_interact_times)
        # get the n-hop neighbors of source and destination nodes
        # three lists to store source nodes' n-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        self.neighbor_sampler.neighbor_sample_from_nodes(src_node_ids_th, node_interact_times_th)
        # Shape: (batch_size, max_input_sequence_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times, src_srcindex_list =  \
            self.neighbor_sampler.get_ret()
        # three lists to store destination nodes' n-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        self.neighbor_sampler.neighbor_sample_from_nodes(dst_node_ids_th, node_interact_times_th)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times, dst_srcindex_list = \
            self.neighbor_sampler.get_ret()

        if not no_time:
            globals.timer.end_neighbor_sample()

        if not no_time:
            globals.timer.start_encodeCo()
        from utils.cpp.src.cpp_cores import count_nodes_cooccurrence
        src_padded_nodes_appearances, dst_padded_nodes_appearances = count_nodes_cooccurrence(src_padded_nodes_neighbor_ids.numpy(), dst_padded_nodes_neighbor_ids.numpy(), 32)
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = torch.from_numpy(src_padded_nodes_appearances).float().to(self.device), torch.from_numpy(dst_padded_nodes_appearances).float().to(self.device)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, cross_edge_neighbor_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
            
        # Tensor, shape (batch_size, dst_max_seq_length*(self.num_high_order_neighbors+1), cross_edge_neighbor_feat_dim)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
            
        if self.hops==2:
            # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
            # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
            src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_pooling(src_padded_nodes_neighbor_co_occurrence_features[:, :max_input_sequence_length, :], src_padded_nodes_neighbor_co_occurrence_features[:, max_input_sequence_length:, :], self.num_high_order_neighbors, 'mean')
            src_padded_nodes_neighbor_co_occurrence_features = self.aggregate_layer(src_padded_nodes_neighbor_co_occurrence_features)
            
            dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_pooling(dst_padded_nodes_neighbor_co_occurrence_features[:, :max_input_sequence_length, :], dst_padded_nodes_neighbor_co_occurrence_features[:, max_input_sequence_length:, :], self.num_high_order_neighbors, 'mean')
            dst_padded_nodes_neighbor_co_occurrence_features = self.aggregate_layer(dst_padded_nodes_neighbor_co_occurrence_features)

        # add identity encoding for the same frequency nodes
        src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids = src_padded_nodes_neighbor_ids.to(self.device), dst_padded_nodes_neighbor_ids.to(self.device)
        src_padded_nodes_edge_ids, dst_padded_nodes_edge_ids = src_padded_nodes_edge_ids.to(self.device), dst_padded_nodes_edge_ids.to(self.device)
        src_padded_nodes_neighbor_times, dst_padded_nodes_neighbor_times = src_padded_nodes_neighbor_times.float().to(self.device), dst_padded_nodes_neighbor_times.float().to(self.device)
        
        if not self.no_id_encode:
            src_padded_nodes_1_neighbor_ids = src_padded_nodes_neighbor_ids[:, :max_input_sequence_length]
            src_neigh_mask = src_padded_nodes_1_neighbor_ids.unsqueeze(1) == src_padded_nodes_1_neighbor_ids.unsqueeze(2)
            src_neigh_mask = src_neigh_mask&(src_padded_nodes_1_neighbor_ids.unsqueeze(1)!=0)
            # src_iden_encode: shape#[batch_size, src_max_seq_length, src_max_seq_length]
            src_iden_encode = src_neigh_mask.float()
            # src_iden_encode = self.compress_layer(src_iden_encode)
            # src_iden_encode = torch.cat([src_iden_encode, torch.zeros_like(src_padded_nodes_neighbor_co_occurrence_features[:,max_input_sequence_length:,:], device=self.device)], dim=1)
            src_padded_nodes_neighbor_co_occurrence_features = torch.cat([src_padded_nodes_neighbor_co_occurrence_features, src_iden_encode], dim=2)
            src_padded_nodes_neighbor_co_occurrence_features = torch.cat([src_padded_nodes_neighbor_co_occurrence_features, torch.zeros([src_padded_nodes_neighbor_co_occurrence_features.shape[0], self.max_input_sequence_length - max_input_sequence_length, src_padded_nodes_neighbor_co_occurrence_features.shape[2]], device=self.device)], dim=1)
            
            # dst_iden_encode: shape#[batch_size, src_max_seq_length, cross_edge_neighbor_feat_dim + src_max_seq_length]
            dst_padded_nodes_1_neighbor_ids = dst_padded_nodes_neighbor_ids[:, :max_input_sequence_length]
            dst_neigh_mask = dst_padded_nodes_1_neighbor_ids.unsqueeze(1) == dst_padded_nodes_1_neighbor_ids.unsqueeze(2)
            dst_neigh_mask = dst_neigh_mask&(dst_padded_nodes_1_neighbor_ids.unsqueeze(1)!=0)
            # dst_iden_encode: shape#[batch_size, dst_max_seq_length, dst_max_seq_length]
            dst_iden_encode = dst_neigh_mask.float()
            # dst_iden_encode = self.compress_layer(dst_iden_encode)
            # dst_iden_encode = torch.cat([dst_iden_encode, torch.zeros_like(dst_padded_nodes_neighbor_co_occurrence_features[:,max_input_sequence_length:,:], device=self.device)], dim=1)
            
            # dst_iden_encode: shape#[batch_size, dst_max_seq_length, cross_edge_neighbor_feat_dim + dst_max_seq_length]
            dst_padded_nodes_neighbor_co_occurrence_features = torch.cat([dst_padded_nodes_neighbor_co_occurrence_features, dst_iden_encode], dim=2)
            dst_padded_nodes_neighbor_co_occurrence_features = torch.cat([dst_padded_nodes_neighbor_co_occurrence_features, torch.zeros([dst_padded_nodes_neighbor_co_occurrence_features.shape[0], self.max_input_sequence_length - max_input_sequence_length, dst_padded_nodes_neighbor_co_occurrence_features.shape[2]], device=self.device)], dim=1)
            
        # src_padded_nodes_neighbor_co_occurrence_features = torch.zeros(src_padded_nodes_neighbor_ids.shape[0], src_padded_nodes_neighbor_ids.shape[1], self.cross_edge_neighbor_feat_dim + self.max_input_sequence_length, device=self.device)
        # dst_padded_nodes_neighbor_co_occurrence_features = torch.zeros(dst_padded_nodes_neighbor_ids.shape[0], dst_padded_nodes_neighbor_ids.shape[1], self.cross_edge_neighbor_feat_dim + self.max_input_sequence_length, device=self.device)
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
        if(self.hops == 2 and not self.no_id_encode):
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
        patches_nodes_neighbor_node_raw_features = torch.cat([src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = torch.cat([src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat([src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_co_occurrence_features = torch.cat([src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features], dim=1)
        
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features]
        patches_data = torch.stack(patches_data, dim=2)
        
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)
        
        if not no_time:
            globals.timer.start_forward()
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        for transformer in self.transformers:
            patches_data = transformer(patches_data)
        if not no_time:
            globals.timer.end_forward()
            
        # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        src_patches_data = patches_data[:, : src_num_patches, :]
        # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        src_patches_data = torch.mean(src_patches_data, dim=1)
        # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer[0](src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer[1](dst_patches_data)

        return src_node_embeddings, dst_node_embeddings
    
    def neighbor_pooling(self, root: torch.Tensor, neighbor:torch.Tensor, neighbor_size: int, aggr:str='mean'):
        # root: shape#[batch_size, num_node, feature_dim neighbor: shape#[batch_size, num_node*neighbor_size, feature_dim]
        assert root.shape[1] * neighbor_size == neighbor.shape[1], "neighbor_pooling size mismatch!!!"
        batch_size = root.shape[0]
        num_node = root.shape[1]
        feature_dim = root.shape[2]
        
        # neighbor: shape#[batch_size, num_node, neighbor_size, feature_dim]
        neighbor = neighbor.reshape(batch_size, num_node, neighbor_size, feature_dim)
        
        # aggregete
        if aggr=='mean':
            non_zero_count = (neighbor != 0).sum(dim=2)
            non_zero_count[non_zero_count==0] = 1
            aggregated_neighbor = neighbor.sum(dim=2)/non_zero_count
            aggregated_result = (root+aggregated_neighbor)/2
        elif aggr=='sum':
            aggregated_neighbor = neighbor.sum(dim=2)
            aggregated_result = root + aggregated_neighbor
        else:
            ValueError('no valid aggregete type!!!')
        
        return aggregated_result

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: torch.Tensor, padded_nodes_edge_ids: torch.Tensor,
                     padded_nodes_neighbor_times: torch.Tensor, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        node_interact_times_tensor = torch.from_numpy(node_interact_times[:, np.newaxis]).float().to(self.device)
        time_differences = node_interact_times_tensor - padded_nodes_neighbor_times
        
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[padded_nodes_neighbor_ids]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[padded_nodes_edge_ids]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=time_differences)

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features.masked_fill_((padded_nodes_neighbor_ids == 0).unsqueeze(-1), 0.0)

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, cross_edge_neighbor_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size
        batch_size = len(padded_nodes_neighbor_node_raw_features)
        
        # Use the unfold method to perform sliding window operations directly on tensors
        def unfold_tensor(tensor, feat_dim):
            return tensor.permute(0, 2, 1).unfold(-1, patch_size, patch_size).permute(0, 2, 3, 1).contiguous().view(batch_size, num_patches, patch_size * feat_dim)

        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = unfold_tensor(padded_nodes_neighbor_node_raw_features, self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = unfold_tensor(padded_nodes_edge_raw_features, self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = unfold_tensor(padded_nodes_neighbor_time_features, self.time_feat_dim)
        if padded_nodes_neighbor_co_occurrence_features is not None:
            patches_nodes_neighbor_co_occurrence_features = unfold_tensor(padded_nodes_neighbor_co_occurrence_features,
                                                                        self.cross_edge_neighbor_feat_dim if self.no_id_encode else self.cross_edge_neighbor_feat_dim+self.max_input_sequence_length//(self.num_high_order_neighbors+1))

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
    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1, num_channels: int = 4, dim_expansion_factor: int = 4):
        """
        TransformerEncoder constructor.

        Args:
            attention_dim (int): The dimension of the attention vectors.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout, batch_first=True)          
            
        self.dropout = nn.Dropout(dropout)
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=dim_expansion_factor * attention_dim),
            nn.Linear(in_features=dim_expansion_factor * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        # Pass through the multi-head attention layer
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        inputs = self.norm_layers[0](inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(query=inputs, key=inputs, value=inputs)[0]
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs
