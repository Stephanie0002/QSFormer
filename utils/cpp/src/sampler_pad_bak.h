#pragma once
#include "head.h"
#include "neighbors.h"
# include "output.h"

class ParallelSampler
{
    public:
        TemporalNeighborBlock& tnb;
        NodeIDType num_nodes;
        EdgeIDType num_edges;
        int threads;
        vector<int> fanouts;
        // vector<NodeIDType> part_ptr;
        // int pid;
        int num_layers;
        string policy;
        std::vector<TemporalGraphBlock> ret;
        int batch_size;

        ParallelSampler(TemporalNeighborBlock& _tnb, NodeIDType _num_nodes, EdgeIDType _num_edges, int _threads, 
                        vector<int>& _fanouts, int _num_layers, string _policy) :
                        tnb(_tnb), num_nodes(_num_nodes), num_edges(_num_edges), threads(_threads), 
                        fanouts(_fanouts), num_layers(_num_layers), policy(_policy)
        {
            omp_set_num_threads(_threads);
            ret.clear();
            ret.resize(_num_layers);
        }

        void reset()
        {
            ret.clear();
            ret.resize(num_layers);
        }

        void neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts, optional<bool> part_unique);
        void neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts);
        void neighbor_sample_from_nodes_with_before_layer(th::Tensor nodes, th::Tensor root_ts, int cur_layer);
        py::tuple get_ret();
        void set_fanouts(vector<int>& _fanouts){
            this->fanouts = _fanouts;
        }
};



void ParallelSampler :: neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts, optional<bool> part_unique)
{
    batch_size = nodes.size(0);
    omp_set_num_threads(threads);
    if(policy == "weighted")
        AT_ASSERTM(tnb.weighted, "Tnb has no weight infomation!");
    else if(policy == "recent")
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
    else if(policy == "uniform")
        ;
    else{
        throw runtime_error("The policy \"" + policy + "\" is not exit!");
    }
    if(tnb.with_timestamp){
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
        AT_ASSERTM(root_ts.has_value(), "Parameter mismatch!");
        neighbor_sample_from_nodes_with_before(nodes, root_ts.value());
    }
    else{
        throw runtime_error("Not support without timestamp!");
    }
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before_layer(
        th::Tensor nodes, th::Tensor root_ts, int cur_layer){
    py::gil_scoped_release release;
    double tot_start_time = omp_get_wtime();
    ret[cur_layer] = TemporalGraphBlock();
    nodes = nodes.view(-1).contiguous();
    root_ts = root_ts.view(-1).contiguous();
    auto nodes_data = get_data_ptr<NodeIDType>(nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);
    int fanout = fanouts[cur_layer];

    long int dim = 1;
    for(int i=0;i<=cur_layer;i++){
        dim *= fanouts[i];
    }
    const size_t array_size = batch_size * dim;
    std::vector<long int> shape = {batch_size, dim};
    ret[cur_layer].sample_nodes = th::zeros({array_size}, th::kInt64);
    ret[cur_layer].eid = th::zeros({array_size}, th::kInt64);
    ret[cur_layer].sample_nodes_ts = th::zeros({array_size}, th::kDouble);

    NodeIDType *nodeArray_ptr = get_data_ptr<NodeIDType>(ret[cur_layer].sample_nodes);
    EdgeIDType *edgeArray_ptr = get_data_ptr<EdgeIDType>(ret[cur_layer].eid);
    TimeStampType *timestampArray_ptr = get_data_ptr<TimeStampType>(ret[cur_layer].sample_nodes_ts);
    
    default_random_engine e(8);//(time(0));
    // double start_time = omp_get_wtime();
    unsigned int time = static_cast<unsigned int>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    unsigned int loc_seed = 0;// tid + time;
// #pragma omp parallel for num_threads(threads) default(shared)
#pragma omp for schedule(static, int(ceil(static_cast<float>(nodes.size(0)) / threads)))
    for(int64_t i=0; i<nodes.size(0); i++){
        // int tid = omp_get_thread_num();
        NodeIDType node = nodes_data[i];
        TimeStampType rtts = ts_data[i];
        if(node==0) continue;//padding data
        
        int end_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), rtts)-tnb.timestamp[node].begin();

        double s_start_time = omp_get_wtime();
        size_t current_batch_offset = i * fanout;
        if ((policy == "recent") || (end_index <= fanout)){
            int start_index = max(0, end_index-fanout);
            std::memcpy(nodeArray_ptr+current_batch_offset, tnb.neighbors[node].data()+start_index, (end_index-start_index)*sizeof(NodeIDType));
            std::memcpy(edgeArray_ptr+current_batch_offset, tnb.eid[node].data()+start_index, (end_index-start_index)*sizeof(EdgeIDType));
            std::memcpy(timestampArray_ptr+current_batch_offset, tnb.timestamp[node].data()+start_index, (end_index-start_index)*sizeof(TimeStampType));
        }
        else{
            //可选邻居边大于扇出的话需要随机选择fanout个邻居
            // uniform_int_distribution<> u(0, end_index-1);
            //cout<<end_index<<endl;
            // cout<<"start:"<<start_index<<" end:"<<end_index<<endl;
            for(int i=0; i<fanout;i++){
                int cid;
                if(policy == "uniform")
                    // cid = u(e);
                    cid = rand_r(&loc_seed) % (end_index);
                else if(policy == "weighted"){
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    cid = sample_multinomial(ew, &loc_seed, end_index);
                }
                nodeArray_ptr[current_batch_offset + i] = tnb.neighbors[node][cid];
                edgeArray_ptr[current_batch_offset + i] = tnb.eid[node][cid];
                timestampArray_ptr[current_batch_offset + i] = tnb.timestamp[node][cid];
            }
        }
        if(tid==0)
            ret[0].sample_time += omp_get_wtime() - s_start_time;
    }
}
    // double end_time = omp_get_wtime();
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;

    ret[cur_layer].sample_nodes.resize_(shape);
    ret[cur_layer].eid.resize_(shape);
    ret[cur_layer].sample_nodes_ts.resize_(shape);

    ret[0].tot_time += omp_get_wtime() - tot_start_time;
    ret[0].sample_edge_num += (ret[cur_layer].eid!=0).sum().item<int64_t>();
    py::gil_scoped_acquire acquire;
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts){
    for(int i=0;i<num_layers;i++){
        if(i==0) neighbor_sample_from_nodes_with_before_layer(nodes, root_ts, i);
        else neighbor_sample_from_nodes_with_before_layer(ret[i-1].sample_nodes, 
                                                          ret[i-1].sample_nodes_ts, i);
    }
}

py::tuple ParallelSampler :: get_ret(){
    auto node_list = py::list();
    auto edge_list = py::list();
    auto ts_list = py::list();
    for(int i=0;i<num_layers;i++){
        node_list.append(ret[i].sample_nodes);
        edge_list.append(ret[i].eid);
        ts_list.append(ret[i].sample_nodes_ts);
    }
    return py::make_tuple(node_list, edge_list, ts_list);
}