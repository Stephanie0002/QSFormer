#include"temporal_neighbor_sampler.h"

TemporalNeighborSampler::TemporalNeighborSampler(
    py::list nodes_neighbor_ids, 
    py::list nodes_edge_ids, 
    py::list nodes_neighbor_times, 
    size_t num_sampled_neighbors
) : num_sampled_neighbors(num_sampled_neighbors) {
    
    assert(nodes_neighbor_ids.size() == nodes_edge_ids.size() && nodes_neighbor_ids.size() == nodes_neighbor_times.size());
    ssize_t num_nodes = nodes_neighbor_ids.size();
    node_neighbors.resize(num_nodes, num_sampled_neighbors);

    for (ssize_t src_node = 0; src_node < num_nodes; ++src_node) {
        const py::array_t<int64_t> node_ids = nodes_neighbor_ids[src_node].cast<py::array_t<int64_t>>();
        const py::array_t<int64_t> edge_ids = nodes_edge_ids[src_node].cast<py::array_t<int64_t>>();
        const py::array_t<float> neighbor_times = nodes_neighbor_times[src_node].cast<py::array_t<float>>();

        ssize_t num_neighbors = node_ids.size();
        const int64_t *node_ids_ptr = node_ids.data();
        const int64_t *edge_ids_ptr = edge_ids.data();
        const float *neighbor_times_ptr = neighbor_times.data();

        for (ssize_t dst_node = 0; dst_node < num_neighbors; ++dst_node) {
            node_neighbors[src_node].addNode(node_ids_ptr[dst_node], edge_ids_ptr[dst_node], neighbor_times_ptr[dst_node]);
        }
    }
}

void TemporalNeighborSampler::reset() {
    // reset each node adjacency list
    for (auto &adj_list : node_neighbors) {
        adj_list.reset();
    }
}

/* Sample recent neighbors for each node in the batch
    * @param node_ids: array of node ids
    * @param timestamps: array of timestamps
    * @return: tuple of () */
py::tuple TemporalNeighborSampler::sample(py::array_t<int64_t> node_ids, py::array_t<float> timestamps) {
    assert(node_ids.size() == timestamps.size());
    
    size_t batch_size = node_ids.size();

    // use numpy arrays directly
    const int64_t *node_ids_ptr = node_ids.data();
    const float *timestamps_ptr = timestamps.data();
    
    size_t max_sequence_length = 0;
    std::vector<AdjacencyNodeIterPair> it_pairs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        const int64_t& node_id = node_ids_ptr[i];
        const float& timestamp = timestamps_ptr[i];
        
        it_pairs[i] = node_neighbors[node_id].sampleNeighbors(timestamp, false /* streaming_mode */);
        size_t sequence_size = it_pairs[i].second - it_pairs[i].first;
        max_sequence_length = std::max(max_sequence_length, sequence_size);
    }
    
    // create output arrays
    const size_t array_size = batch_size * max_sequence_length;
    auto nodeArray = py::array_t<int64_t>(array_size);
    nodeArray.resize({batch_size, max_sequence_length});
    auto edgeArray = py::array_t<int64_t>(array_size);
    edgeArray.resize({batch_size, max_sequence_length});
    auto timestampArray = py::array_t<float>(array_size);
    timestampArray.resize({batch_size, max_sequence_length});
    
    int64_t *nodeArray_ptr = nodeArray.mutable_data();
    memset(nodeArray_ptr, 0, array_size * sizeof(int64_t));
    int64_t *edgeArray_ptr = edgeArray.mutable_data();
    memset(edgeArray_ptr, 0, array_size * sizeof(int64_t));
    float *timestampArray_ptr = timestampArray.mutable_data();
    memset(timestampArray_ptr, 0, array_size * sizeof(float));

    for (size_t i = 0; i < batch_size; ++i) {
        const AdjacencyNodeIterPair& it_pair = it_pairs[i];
        auto sample_begin = it_pair.first;
        auto sample_end = it_pair.second;

        size_t current_sequence_size = 0;
        size_t current_batch_offset = i * max_sequence_length;
        for (auto it = sample_begin; it != sample_end; ++it) {
            nodeArray_ptr[current_batch_offset + current_sequence_size] = it->node_id;
            edgeArray_ptr[current_batch_offset + current_sequence_size] = it->edge_id;
            timestampArray_ptr[current_batch_offset + current_sequence_size] = it->timestamp;
            ++current_sequence_size;
        }
    }
    
    return py::make_tuple(nodeArray, edgeArray, timestampArray);
}

/* Sample recent neighbors for each node in the batch
    * @param src_node_ids: array of src node ids
    * @param dst_node_ids: array of dst node ids
    * @param timestamps: array of timestamps
    * @return: tuple of () */
py::tuple TemporalNeighborSampler::sampleAndPadding(py::array_t<int64_t> src_node_ids, py::array_t<int64_t> dst_node_ids, py::array_t<float> timestamps, size_t patch_size, bool streaming_mode) {
    size_t batch_size = timestamps.size();

    // use numpy arrays directly
    const int64_t *src_node_ids_ptr = src_node_ids.data();
    const int64_t *dst_node_ids_ptr = dst_node_ids.data();
    const float *timestamps_ptr = timestamps.data();
    
    size_t src_max_sequence_length = 0;
    size_t dst_max_sequence_length = 0;
    std::pair<AdjacencyNodeIterPairList, AdjacencyNodeIterPairList> it_pairs = sampleNeighbors(
        src_node_ids_ptr, dst_node_ids_ptr, timestamps_ptr, 
        batch_size, src_max_sequence_length, dst_max_sequence_length,
        streaming_mode
    );

    AdjacencyNodeIterPairList& src_it_pairs = it_pairs.first;
    AdjacencyNodeIterPairList& dst_it_pairs = it_pairs.second;
    
    return py::make_tuple(
        paddingSequences(src_node_ids_ptr, timestamps_ptr, src_it_pairs, batch_size, patch_size, src_max_sequence_length),
        paddingSequences(dst_node_ids_ptr, timestamps_ptr, dst_it_pairs, batch_size, patch_size, dst_max_sequence_length)
    );            
}

void TemporalNeighborSampler::printAdjacencyList(size_t node_id) const {
    std::cout << "node " << node_id << ": \n";
    std::cout << node_neighbors[node_id].toString() << std::endl;
}

void TemporalNeighborSampler::printAdjacencyLists() const {
    for (size_t i = 1; i < node_neighbors.size(); ++i) {
        printAdjacencyList(i);
    }
}

// sample neighbors and find out max sequence length
std::pair<AdjacencyNodeIterPairList, AdjacencyNodeIterPairList>
TemporalNeighborSampler::sampleNeighbors(
    const int64_t *src_node_ids_ptr,
    const int64_t *dst_node_ids_ptr,
    const float *timestamps_ptr,
    const size_t& batch_size,
    size_t& src_max_sequence_length,
    size_t& dst_max_sequence_length,
    const bool streaming_mode
) {
    // Initialize lengths to zero
    src_max_sequence_length = dst_max_sequence_length = 0;

    std::vector<AdjacencyNodeIterPair> src_it_pairs(batch_size);
    std::vector<AdjacencyNodeIterPair> dst_it_pairs(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        // current event information
        const int64_t& src_node_id = src_node_ids_ptr[i];
        const int64_t& dst_node_id = dst_node_ids_ptr[i];
        const float& timestamp = timestamps_ptr[i];
        
        src_it_pairs[i] = node_neighbors[src_node_id].sampleNeighbors(timestamp, streaming_mode);
        const size_t src_sequence_length = std::distance(src_it_pairs[i].first, src_it_pairs[i].second);
        src_max_sequence_length = std::max(src_max_sequence_length, src_sequence_length);

        dst_it_pairs[i] = node_neighbors[dst_node_id].sampleNeighbors(timestamp, streaming_mode);
        const size_t dst_sequence_length = std::distance(dst_it_pairs[i].first, dst_it_pairs[i].second);
        dst_max_sequence_length = std::max(dst_max_sequence_length, dst_sequence_length);
    }

    return std::make_pair(src_it_pairs, dst_it_pairs);
}

py::tuple TemporalNeighborSampler::paddingSequences(
    const int64_t *node_ids_ptr, 
    const float *timestamps_ptr,
    const std::vector<AdjacencyNodeIterPair>& it_pairs,
    const size_t& batch_size,
    const size_t& patch_size,
    const size_t& max_sequence_length
) {
    // Adjust the max_sequence_length based on patch_size and include the target node itself
    size_t padded_sequence_size = max_sequence_length + 1; // Adding 1 for the target node itself
    padded_sequence_size += -padded_sequence_size % patch_size; // Ensure max_sequence_length is a multiple of patch_size
    
    // Pre-allocating three ndarrays with shape (batch_size, max_seq_length)
    const size_t array_size = batch_size * padded_sequence_size;

    // Allocate NumPy arrays, initialized to zero
    auto nodeArray = py::array_t<int64_t>({batch_size, padded_sequence_size}, 0);
    auto edgeArray = py::array_t<int64_t>({batch_size, padded_sequence_size}, 0);
    auto timestampArray = py::array_t<float>({batch_size, padded_sequence_size}, 0);

    // initialize arrays
    int64_t *nodeArray_ptr = nodeArray.mutable_data();
    memset(nodeArray_ptr, 0, array_size * sizeof(int64_t));
    int64_t *edgeArray_ptr = edgeArray.mutable_data();
    memset(edgeArray_ptr, 0, array_size * sizeof(int64_t));
    float *timestampArray_ptr = timestampArray.mutable_data();
    memset(timestampArray_ptr, 0, array_size * sizeof(float));

    // Loop through each list and update the pre-allocated arrays accordingly
    for (size_t i = 0; i < batch_size; ++i) {
        const AdjacencyNodeIterPair& it_pair = it_pairs[i];
        auto sample_begin = it_pair.first;
        auto sample_end = it_pair.second;

        // Calculate the current batch offset
        size_t current_batch_offset = i * padded_sequence_size;

        // Setting the target node and its interact time for each sequence
        nodeArray_ptr[current_batch_offset + 0] = node_ids_ptr[i];
        timestampArray_ptr[current_batch_offset + 0] = timestamps_ptr[i];

        size_t current_sequence_size = 1;
        for (auto it = sample_begin; it != sample_end; ++it) {
            nodeArray_ptr[current_batch_offset + current_sequence_size] = it->node_id;
            edgeArray_ptr[current_batch_offset + current_sequence_size] = it->edge_id;
            timestampArray_ptr[current_batch_offset + current_sequence_size] = it->timestamp;
            ++current_sequence_size;
        }
    }
    
    // three ndarrays with shape (batch_size, max_seq_length)
    return py::make_tuple(nodeArray, edgeArray, timestampArray);
}