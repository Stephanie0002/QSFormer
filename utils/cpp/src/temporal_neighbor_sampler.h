#include "adjacency_list.h"

#include <list>
#include <vector>
#include <cassert>
#include <iostream>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class TemporalNeighborSampler {
public:
    TemporalNeighborSampler(
        py::list nodes_neighbor_ids, 
        py::list nodes_edge_ids, 
        py::list nodes_neighbor_times, 
        size_t num_sampled_neighbors = 31
    );

    void reset();
    py::tuple sample(py::array_t<int64_t> node_ids, py::array_t<float> timestamps);
    py::tuple sampleAndPadding(py::array_t<int64_t> src_node_ids, py::array_t<int64_t> dst_node_ids, py::array_t<float> timestamps, size_t patch_size = 1, bool streaming_mode = false);
    void printAdjacencyList(size_t node_id) const;
    void printAdjacencyLists() const;

private:
    size_t num_sampled_neighbors;
    std::vector<AdjacencyList> node_neighbors;

    struct NeighborInfo {
        size_t max_sequence_length;
        std::vector<AdjacencyNodeIterPair> it_pairs;

        NeighborInfo(size_t batch_size) : max_sequence_length(0), it_pairs(batch_size) {}
    };

    std::pair<AdjacencyNodeIterPairList, AdjacencyNodeIterPairList>
    sampleNeighbors(
        const int64_t *src_node_ids_ptr,
        const int64_t *dst_node_ids_ptr,
        const float *timestamps_ptr,
        const size_t& batch_size,
        size_t& src_max_sequence_length,
        size_t& dst_max_sequence_length,
        const bool streaming_mode = false
    );

    py::tuple paddingSequences(
        const int64_t *node_ids_ptr, 
        const float *timestamps_ptr,
        const std::vector<AdjacencyNodeIterPair>& it_pairs,
        const size_t& batch_size,
        const size_t& patch_size,
        const size_t& max_sequence_length
    );
};