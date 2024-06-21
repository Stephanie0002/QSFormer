#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <algorithm>

struct AdjacencyNode {
    int64_t node_id;
    int64_t edge_id;
    float timestamp;

    AdjacencyNode(int64_t node_id, int64_t edge_id, float timestamp)
        : node_id(node_id), edge_id(edge_id), timestamp(timestamp) {}
    
    bool operator<(const AdjacencyNode& other) const {
        // return timestamp < other.timestamp;
        return std::tie(timestamp, node_id, edge_id) < std::tie(other.timestamp, other.node_id, other.edge_id);
    }

    std::string toString() const {
        return "AdjacencyNode("
            " NodeId: " + std::to_string(node_id) + 
            " EdgeId: " + std::to_string(edge_id) + 
            " Timestamp: " + std::to_string(timestamp)+ ")";
    }
};

using AdjacencyNodeIterPair = std::pair<
    std::vector<AdjacencyNode>::iterator, 
    std::vector<AdjacencyNode>::iterator
>;

using AdjacencyNodeIterPairList = std::vector<AdjacencyNodeIterPair>;

class AdjacencyList {
    public:
        AdjacencyList(size_t num_sampled_neighbors)
            : current_index(0), num_sampled_neighbors(num_sampled_neighbors) {}

        void reset() {
            current_index = 0;
        }

        void addNode(int64_t node_id, int64_t edge_id, float timestamp) {
            adjacency_list.emplace_back(AdjacencyNode(node_id, edge_id, timestamp));
        }

        void sort() {
            std::sort(adjacency_list.begin(), adjacency_list.end());
        }

        AdjacencyNodeIterPair sampleNeighbors(float timestamp, bool streaming_mode = false) {
            if (streaming_mode) {
                /* As we are not sure whether the node to be sampled belongs to a positive edge or not, 
                * we could not enable streaming sample for all situations. */
                while (current_index < adjacency_list.size() && adjacency_list[current_index].timestamp < timestamp) {
                    ++current_index;
                }

                size_t start_index = current_index > num_sampled_neighbors ? current_index - num_sampled_neighbors : 0;

                auto sample_begin = adjacency_list.begin() + start_index;
                auto sample_end = adjacency_list.begin() + current_index;

                return std::make_pair(sample_begin, sample_end);
            } else {
                /* Use binary search to find the indices into a sorted array 
                * such that, if the corresponding elements in v were inserted before the indices, 
                * the order of a would be preserved. */
                auto binary_sample_end = std::lower_bound(adjacency_list.begin(), adjacency_list.end(), timestamp, 
                    [](const AdjacencyNode& node, float timestamp) {
                        return node.timestamp < timestamp;
                    }
                );

                auto binary_sample_begin = (size_t)(binary_sample_end - adjacency_list.begin()) > num_sampled_neighbors 
                    ? binary_sample_end - num_sampled_neighbors 
                    : adjacency_list.begin();
                
                return std::make_pair(binary_sample_begin, binary_sample_end);
            }
        }

        std::string toString() const {
            std::string str = "AdjacencyList(";
            for (const auto& node : adjacency_list) {
                str += "\n\t" + node.toString() + ", ";
            }
            return str + "\n)";
        }

    private:

        size_t current_index;

        size_t num_sampled_neighbors;

        std::vector<AdjacencyNode> adjacency_list;
};