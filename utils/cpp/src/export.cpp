#include "co_counter.h"
#include "sampler.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m
    .def("count_nodes_cooccurrence", &countNodesCooccurence, "Function to count node co-occurrences")
    .def("get_neighbors",  &get_neighbors, py::return_value_policy::reference);

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
}