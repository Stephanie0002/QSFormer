#pragma once

#include <vector>
#include <cassert>
#include <utility>
#include <iostream>

#include "head.h"

void countFrequencies(phmap::parallel_flat_hash_map<int64_t, int64_t>& freqMap, const int64_t *input_ptr, const ssize_t input_size) {
    // count node frequency
    for (ssize_t i = 0; i < input_size; ++i) {
        if(input_ptr[i]!=0)
            freqMap[input_ptr[i]]++;
    }
}

/**
 * @brief Count node co-occurrence in a batch of graphs
 * @param batch_src_nodes: (batch_size, src_seq_length)
 * @param batch_dst_nodes: (batch_size, dst_seq_length)
*/
py::tuple countNodesCooccurence(py::array_t<int64_t>& batch_src_nodes, py::array_t<int64_t>& batch_dst_nodes, int threads) {
    assert(batch_src_nodes.ndim() == 2 && batch_dst_nodes.ndim() == 2);

    assert(batch_src_nodes.shape(0) == batch_dst_nodes.shape(0));
    ssize_t batch_size = batch_src_nodes.shape(0);

    // src count array: (batch_size, seq_length, 2)
    ssize_t src_seq_len = batch_src_nodes.shape(1);
    auto py_src_counts = py::array_t<int64_t>({batch_size, src_seq_len, (ssize_t) 2});
    int64_t *py_src_counts_ptr = py_src_counts.mutable_data();
    
    // dst count array: (batch_size, seq_length, 2)
    ssize_t dst_seq_len = batch_dst_nodes.shape(1);
    auto py_dst_counts = py::array_t<int64_t>({batch_size, dst_seq_len, (ssize_t) 2});
    int64_t *py_dst_counts_ptr = py_dst_counts.mutable_data();
    
    int64_t *batch_src_nodes_ptr = batch_src_nodes.mutable_data();
    int64_t *batch_dst_nodes_ptr = batch_dst_nodes.mutable_data();
#pragma omp parallel for num_threads(threads) default(shared)
    for (ssize_t i = 0; i < batch_size; ++i) {
        // batch_src_nodes[i], batch_dst_nodes[i]
        int64_t *src_nodes_ptr = batch_src_nodes_ptr + i * src_seq_len;
        int64_t *dst_nodes_ptr = batch_dst_nodes_ptr + i * dst_seq_len;

        // Count node frequencies
        phmap::parallel_flat_hash_map<int64_t, int64_t> src_map(src_seq_len>>4), dst_map(dst_seq_len>>4);
        countFrequencies(src_map, src_nodes_ptr, src_seq_len);
        countFrequencies(dst_map, dst_nodes_ptr, dst_seq_len);

        ssize_t src_batch_offset = i * src_seq_len * 2;
        for (ssize_t node = 0; node < src_seq_len; ++node) {
            const int64_t& src_node = src_nodes_ptr[node];
            auto src_it = src_map.find(src_node);
            auto dst_it = dst_map.find(src_node);

            ssize_t seq_offset = src_batch_offset + node * 2;
            py_src_counts_ptr[seq_offset] = src_it != src_map.end() ? src_it->second : 0;
            py_src_counts_ptr[seq_offset | 1] = dst_it != dst_map.end() ? dst_it->second : 0;
        }

        ssize_t dst_batch_offset = i * dst_seq_len * 2;
        for (ssize_t node = 0; node < dst_seq_len; ++node) {
            const int64_t& dst_node = dst_nodes_ptr[node];
            auto src_it = src_map.find(dst_node);
            auto dst_it = dst_map.find(dst_node);

            ssize_t seq_offset = dst_batch_offset + node * 2;
            py_dst_counts_ptr[seq_offset] = src_it != src_map.end() ? src_it->second : 0;
            py_dst_counts_ptr[seq_offset | 1] = dst_it != dst_map.end() ? dst_it->second : 0;
        }
    }

    return py::make_tuple(py_src_counts, py_dst_counts);
}




// /**
//  * @brief Count node co-occurrence in a batch of graphs
//  * @param batch_src_nodes: (batch_size, src_seq_length)
//  * @param batch_dst_nodes: (batch_size, dst_seq_length)
// */
// py::tuple countNodesCooccurence(th::Tensor batch_src_nodes, th::Tensor batch_dst_nodes, int threads) {
//     assert(batch_src_nodes.ndim() == 2 && batch_dst_nodes.ndim() == 2);
//     assert(batch_src_nodes.size(0) == batch_dst_nodes.size(0));
//     ssize_t batch_size = batch_src_nodes.size(0);

//     // src count array: (batch_size, seq_length, 2)
//     ssize_t src_seq_len = batch_src_nodes.size(1);
//     auto py_src_counts = th::zeros({batch_size, src_seq_len, 2}, th::kInt64);
//     int64_t *py_src_counts_ptr = get_data_ptr<int64_t>(py_src_counts);
    
//     // dst count array: (batch_size, seq_length, 2)
//     ssize_t dst_seq_len = batch_dst_nodes.size(1);
//     auto py_dst_counts = th::zeros({batch_size, dst_seq_len, 2}, th::kInt64);
//     int64_t *py_dst_counts_ptr = get_data_ptr<int64_t>(py_dst_counts);
    
//     int64_t *batch_src_nodes_ptr = get_data_ptr<int64_t>(batch_src_nodes);
//     int64_t *batch_dst_nodes_ptr = get_data_ptr<int64_t>(batch_dst_nodes);
// #pragma omp parallel for num_threads(threads) default(shared)
//     for (ssize_t i = 0; i < batch_size; ++i) {
//         // batch_src_nodes[i], batch_dst_nodes[i]
//         int64_t *src_nodes_ptr = batch_src_nodes_ptr + i * src_seq_len;
//         int64_t *dst_nodes_ptr = batch_dst_nodes_ptr + i * dst_seq_len;
//         //src count apprearance
//         ssize_t src_batch_offset = i * src_seq_len * 2;
//         for (ssize_t j = 0; j < src_seq_len; j++) {
//             if(src_nodes_ptr[j]!=0){
//                 ssize_t seq_offset = src_batch_offset + j * 2;
//                 py_src_counts_ptr[seq_offset]++;
//                 py_dst_counts_ptr[seq_offset | 1]++;
//             }
//         }
//         //dst count apprearance
//         ssize_t dst_batch_offset = i * dst_seq_len * 2;
//         for (ssize_t j = 0; j < dst_seq_len; j++) {
//             if(dst_nodes_ptr[j]!=0){
//                 ssize_t seq_offset = dst_batch_offset + j * 2;
//                 py_dst_counts_ptr[seq_offset]++;
//                 py_src_counts_ptr[seq_offset | 1]++;
//             }
//         }
//     }
//     return py::make_tuple(py_src_counts, py_dst_counts);
// }