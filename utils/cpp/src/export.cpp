#include "temporal_neighbor_sampler.h"
#include "co_counter.h"
#include "tppr.h"
#include "sampler.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m
    .def("count_nodes_cooccurrence", &countNodesCooccurence, "Function to count node co-occurrences")
    .def("get_neighbors",  &get_neighbors, py::return_value_policy::reference);

    py::class_<TemporalNeighborSampler>(m, "TemporalNeighborSampler", py::module_local())
        .def(py::init<py::list, py::list, py::list, size_t>())
        .def("reset", &TemporalNeighborSampler::reset)
        .def("sample_neighbors", &TemporalNeighborSampler::sample)
        .def("sample_and_padding_neighbors", &TemporalNeighborSampler::sampleAndPadding)
        .def("print_adjacency_list", &TemporalNeighborSampler::printAdjacencyList)
        .def("print_adjacency_lists", &TemporalNeighborSampler::printAdjacencyLists);

    py::class_<ParallelSampler>(m, "ParallelSampler")
        .def(py::init<TemporalNeighborBlock &, NodeIDType, EdgeIDType, int,
                      vector<int>&, string>())
        .def_readonly("ret", &ParallelSampler::ret, py::return_value_policy::reference)
        .def_readonly("sample_neighbor_strategy", &ParallelSampler::policy, py::return_value_policy::reference)
        .def("neighbor_sample_from_nodes", &ParallelSampler::neighbor_sample_from_nodes)
        // .def("reset", &ParallelSampler::reset)
        .def("get_ret", &ParallelSampler::get_ret)
        .def("set_fanouts", &ParallelSampler::set_fanouts);

    py::class_<TemporalNeighborBlock>(m, "TemporalNeighborBlock")
        .def(py::init<vector<vector<NodeIDType>>&, 
                      vector<int64_t> &>())
        .def(py::pickle(
            [](const TemporalNeighborBlock& tnb) { return tnb.serialize(); },
            [](const std::string& s) { return TemporalNeighborBlock::deserialize(s); }
        ))
        .def("update_neighbors_with_time", 
            &TemporalNeighborBlock::update_neighbors_with_time)
        .def("update_edge_weight", 
            &TemporalNeighborBlock::update_edge_weight)
        .def("update_node_weight", 
            &TemporalNeighborBlock::update_node_weight)
        .def("update_all_node_weight", 
            &TemporalNeighborBlock::update_all_node_weight)
        .def_readonly("neighbors", &TemporalNeighborBlock::neighbors, py::return_value_policy::reference)
        .def_readonly("timestamp", &TemporalNeighborBlock::timestamp, py::return_value_policy::reference)
        .def_readonly("edge_weight", &TemporalNeighborBlock::edge_weight, py::return_value_policy::reference)
        .def_readonly("eid", &TemporalNeighborBlock::eid, py::return_value_policy::reference)
        .def_readonly("deg", &TemporalNeighborBlock::deg, py::return_value_policy::reference)
        .def_readonly("with_eid", &TemporalNeighborBlock::with_eid, py::return_value_policy::reference)
        .def_readonly("with_timestamp", &TemporalNeighborBlock::with_timestamp, py::return_value_policy::reference)
        .def_readonly("weighted", &TemporalNeighborBlock::weighted, py::return_value_policy::reference);
    
    py::class_<ParallelTppRComputer>(m, "ParallelTppRComputer")
        .def(py::init<TemporalNeighborBlock &, NodeIDType, EdgeIDType, int,
                      int, int, vector<float>&, vector<float>& >())
        .def_readonly("ret", &ParallelTppRComputer::ret, py::return_value_policy::reference)
        .def("reset_ret", &ParallelTppRComputer::reset_ret)
        .def("reset_tppr", &ParallelTppRComputer::reset_tppr)
        .def("reset_val_tppr", &ParallelTppRComputer::reset_val_tppr)
        .def("backup_tppr", &ParallelTppRComputer::backup_tppr)
        .def("restore_tppr", &ParallelTppRComputer::restore_tppr)
        .def("restore_val_tppr", &ParallelTppRComputer::restore_val_tppr)
        .def("get_pruned_topk", &ParallelTppRComputer::get_pruned_topk)
        .def("extract_streaming_tppr", &ParallelTppRComputer::extract_streaming_tppr)
        .def("streaming_topk", &ParallelTppRComputer::streaming_topk)
        .def("single_streaming_topk", &ParallelTppRComputer::single_streaming_topk)
        .def("streaming_topk_no_fake", &ParallelTppRComputer::streaming_topk_no_fake)
        .def("compute_val_tppr", &ParallelTppRComputer::compute_val_tppr)
        .def("get_ret", [](const ParallelTppRComputer &pt) { return pt.ret; });
}