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
        int num_layers;// 1 or 2
        string policy;
        TemporalGraphBlock ret;
        int batch_size;

        ParallelSampler(TemporalNeighborBlock& _tnb, NodeIDType _num_nodes, EdgeIDType _num_edges, int _threads, 
                        vector<int>& _fanouts, string _policy) :
                        tnb(_tnb), num_nodes(_num_nodes), num_edges(_num_edges), threads(_threads), 
                        fanouts(_fanouts), policy(_policy)
        {
            omp_set_num_threads(_threads);
            num_layers = _fanouts.size();
        }

        // void reset()
        // {
        // }

        void neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts);
        void neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts);
        void neighbor_sample_from_nodes_with_before_layer_0(th::Tensor nodes, th::Tensor root_ts);
        void neighbor_sample_from_nodes_with_before_layer_1();
        py::tuple get_ret();
        void set_fanouts(vector<int>& _fanouts){
            this->fanouts = _fanouts;
            this->num_layers = _fanouts.size();
        }
};

void ParallelSampler :: neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts)
{
    batch_size = nodes.size(0);

    ret = TemporalGraphBlock();
    int array_size = num_layers==2 ? fanouts[0]*(fanouts[1]+1) : fanouts[0];
    ret.sample_nodes = th::zeros({batch_size, array_size}, th::kInt64);
    ret.eid = th::zeros({batch_size, array_size}, th::kInt64);
    ret.sample_nodes_ts = th::zeros({batch_size, array_size}, th::kDouble);
    ret.src_index = th::zeros({batch_size, array_size}, th::kInt64);

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

void ParallelSampler :: neighbor_sample_from_nodes_with_before_layer_0(th::Tensor nodes, th::Tensor root_ts){
    py::gil_scoped_release release;
    double tot_start_time = omp_get_wtime();
    int fanout = fanouts[0];
    nodes = nodes.contiguous();
    root_ts = root_ts.contiguous();
    auto nodes_data = get_data_ptr<NodeIDType>(nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);

    NodeIDType *nodeArray_ptr = get_data_ptr<NodeIDType>(ret.sample_nodes);
    EdgeIDType *edgeArray_ptr = get_data_ptr<EdgeIDType>(ret.eid);
    TimeStampType *timestampArray_ptr = get_data_ptr<TimeStampType>(ret.sample_nodes_ts);
    NodeIDType *srcArray_ptr = get_data_ptr<NodeIDType>(ret.src_index);
    int dim = ret.sample_nodes.size(1);

    // double start_time = omp_get_wtime();
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    unsigned int loc_seed = 0 + tid;// tid + time;
// #pragma omp parallel for num_threads(threads) default(shared)
#pragma omp for schedule(static, int(ceil(static_cast<float>(batch_size) / threads)))
    for(int64_t i=0; i<batch_size; i++){
        // int tid = omp_get_thread_num();    
        NodeIDType node = nodes_data[i];
        TimeStampType rtts = ts_data[i];
        if(node==0) continue;//padding data
        std::fill(srcArray_ptr+i*dim, srcArray_ptr+i*dim+fanout, 1);
        
        int end_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), rtts)-tnb.timestamp[node].begin();

        double s_start_time = omp_get_wtime();
        if ((policy == "recent") || (end_index <= fanout)){
            int start_index = max(0, end_index-fanout);
            int len = end_index-start_index;

            std::memcpy(nodeArray_ptr+i*dim+fanout-len, tnb.neighbors[node].data()+start_index, len*sizeof(NodeIDType));
            std::memcpy(edgeArray_ptr+i*dim+fanout-len, tnb.eid[node].data()+start_index, len*sizeof(EdgeIDType));
            std::memcpy(timestampArray_ptr+i*dim+fanout-len, tnb.timestamp[node].data()+start_index, len*sizeof(TimeStampType));
        }
        else{
            //可选邻居边大于扇出的话需要随机选择fanout个邻居
            // uniform_int_distribution<> u(0, end_index-1);
            //cout<<end_index<<endl;
            // cout<<"start:"<<start_index<<" end:"<<end_index<<endl;
            for(int j=0; j<fanout;j++){
                int cid;
                if(policy == "uniform")
                    // cid = u(e);
                    cid = rand_r(&loc_seed) % (end_index);
                else if(policy == "weighted"){
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    cid = sample_multinomial(ew, &loc_seed, end_index);
                }
                int offset = i*dim+j;
                nodeArray_ptr[offset] = tnb.neighbors[node][cid];
                edgeArray_ptr[offset] = tnb.eid[node][cid];
                timestampArray_ptr[offset] = tnb.timestamp[node][cid];
                srcArray_ptr[offset] = 1;
            }
        }
    }
}
    py::gil_scoped_acquire acquire;
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before_layer_1(){
    py::gil_scoped_release release;
    double tot_start_time = omp_get_wtime();
    int fanout = fanouts[1];
    
    NodeIDType *nodeArray_ptr = get_data_ptr<NodeIDType>(ret.sample_nodes);
    EdgeIDType *edgeArray_ptr = get_data_ptr<EdgeIDType>(ret.eid);
    TimeStampType *timestampArray_ptr = get_data_ptr<TimeStampType>(ret.sample_nodes_ts);
    // NodeIDType *srcArray_ptr = get_data_ptr<NodeIDType>(ret.src_index);
    int dim = ret.sample_nodes.size(1);

    // double start_time = omp_get_wtime();
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    unsigned int loc_seed = 0 + tid;// tid + time;
// #pragma omp parallel for num_threads(threads) default(shared)
#pragma omp for schedule(static, int(ceil(static_cast<float>(batch_size) / threads)))
    for(int64_t i=0; i<batch_size; i++){
        for(int64_t k=0;k<fanouts[0];k++){
            // int tid = omp_get_thread_num();
            NodeIDType node = nodeArray_ptr[i*dim+k];
            TimeStampType rtts = timestampArray_ptr[i*dim+k];
            if(node==0) continue;//padding data
            
            int end_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), rtts)-tnb.timestamp[node].begin();

            double s_start_time = omp_get_wtime();
            if ((policy == "recent") || (end_index <= fanout)){
                int start_index = max(0, end_index-fanout);
                int len = end_index-start_index;
                std::memcpy(nodeArray_ptr+i*dim+fanouts[0]+fanout*(k+1)-len, tnb.neighbors[node].data()+start_index, len*sizeof(NodeIDType));
                std::memcpy(edgeArray_ptr+i*dim+fanouts[0]+fanout*(k+1)-len, tnb.eid[node].data()+start_index, len*sizeof(EdgeIDType));
                std::memcpy(timestampArray_ptr+i*dim+fanouts[0]+fanout*(k+1)-len, tnb.timestamp[node].data()+start_index, len*sizeof(TimeStampType));
                // std::fill(srcArray_ptr+i*dim+fanouts[0]+fanout*(k+1)-len, srcArray_ptr+i*dim+fanouts[0]+fanout*(k+1), node);
            }
            else{
                //可选邻居边大于扇出的话需要随机选择fanout个邻居
                for(int j=0; j<fanout;j++){
                    int cid;
                    if(policy == "uniform")
                        // cid = u(e);
                        cid = rand_r(&loc_seed) % (end_index);
                    else if(policy == "weighted"){
                        const vector<WeightType>& ew = tnb.edge_weight[node];
                        cid = sample_multinomial(ew, &loc_seed, end_index);
                    }
                    int offset = i*dim+fanouts[0]+fanout*k+j;
                    nodeArray_ptr[offset] = tnb.neighbors[node][cid];
                    edgeArray_ptr[offset] = tnb.eid[node][cid];
                    timestampArray_ptr[offset] = tnb.timestamp[node][cid];
                    // srcArray_ptr[offset] = node;
                }
            }
        }
    }
}
    py::gil_scoped_acquire acquire;
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts){
    neighbor_sample_from_nodes_with_before_layer_0(nodes, root_ts);
    if(num_layers == 2)
        neighbor_sample_from_nodes_with_before_layer_1();
}

py::tuple ParallelSampler :: get_ret(){
    return py::make_tuple(ret.sample_nodes, ret.eid, ret.sample_nodes_ts, ret.src_index);
}