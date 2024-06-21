#pragma once
#include <iostream>
#include <algorithm>
#include <numeric>
#include <torch/extension.h>
#include <omp.h>
#include <time.h>
#include <random>
#include "parallel_hashmap/phmap.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;
namespace th = torch;

typedef int64_t NodeIDType;
typedef int64_t EdgeIDType;
typedef float WeightType;
typedef double TimeStampType;
typedef tuple<NodeIDType, EdgeIDType, TimeStampType> PPRKeyType;
typedef double PPRValueType;
typedef phmap::parallel_flat_hash_map<PPRKeyType, PPRValueType> PPRDictType;
typedef vector<PPRDictType> PPRListDictType;
typedef vector<vector<PPRDictType>> PPRListListDictType;
typedef vector<vector<double>> NormListType;

class TemporalNeighborBlock;
class TemporalGraphBlock;
class ParallelSampler;

TemporalNeighborBlock& get_neighbors(string graph_name, th::Tensor row, th::Tensor col, int64_t num_nodes, int is_distinct, optional<th::Tensor> eid, optional<th::Tensor> edge_weight, optional<th::Tensor> time);
th::Tensor heads_unique(th::Tensor array, th::Tensor heads, int threads);
int nodeIdToInOut(NodeIDType nid, int pid, const vector<NodeIDType>& part_ptr);
int nodeIdToPartId(NodeIDType nid, const vector<NodeIDType>& part_ptr);
vector<th::Tensor> divide_nodes_to_part(th::Tensor nodes, const vector<NodeIDType>& part_ptr, int threads);
NodeIDType sample_multinomial(const vector<WeightType>& weights, unsigned int* loc_seed, NodeIDType end_index);
vector<int64_t> sample_max(const vector<WeightType>& weights, int k);



// 辅助函数
template<typename T>
inline py::array vec2npy(const std::vector<T> &vec)
{
    // need to let python garbage collector handle C++ vector memory 
    // see https://github.com/pybind/pybind11/issues/1042
    // non-copy value transfer
    auto v = new std::vector<T>(vec);
    auto capsule = py::capsule(v, [](void *v)
                               { delete reinterpret_cast<std::vector<T> *>(v); });
    return py::array(v->size(), v->data(), capsule);
    // return py::array(vec.size(), vec.data());
}

template <typename T>
T* get_data_ptr(const th::Tensor& tensor) {
    AT_ASSERTM(tensor.is_contiguous(), "Offset tensor must be contiguous");
    // AT_ASSERTM(tensor.dim() == 1, "Offset tensor must be one-dimensional");
    return tensor.data_ptr<T>();
}

template <typename T>
vector<T*> get_list_data_ptr(const vector<th::Tensor>& tensor) {
    vector<T*> ret(tensor.size());
    for(auto t:tensor){
        AT_ASSERTM(t.is_contiguous(), "Offset tensor must be contiguous");
        // AT_ASSERTM(t.dim() == 1, "Offset tensor must be one-dimensional");
        ret.emplace_back(t.data_ptr<T>());
    }
    return ret;
}

template <typename T>
th::Tensor vecToTensor(const std::vector<T>& vec) {
    // 确定数据类型
    th::ScalarType dtype;
    if (std::is_same<T, int64_t>::value) {
        dtype = th::kInt64;
    } else if (std::is_same<T, float>::value) {
        dtype = th::kFloat32;
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    // 创建Tensor
    th::Tensor tensor = th::from_blob(
        const_cast<T*>(vec.data()), /* 数据指针 */
        {static_cast<long>(vec.size())}, /* 尺寸 */
        dtype /* 数据类型 */
    );

    return tensor;//.clone(); // 克隆Tensor以拷贝数据
}

/*-------------------------------------------------------------------------------------**
**------------Utils--------------------------------------------------------------------**
**-------------------------------------------------------------------------------------*/

th::Tensor heads_unique(th::Tensor array, th::Tensor heads, int threads){
    auto array_ptr = array.data_ptr<NodeIDType>();
    phmap::parallel_flat_hash_set<NodeIDType> s(array_ptr, array_ptr+array.numel());
    if(heads.numel()==0) return th::tensor(vector<NodeIDType>(s.begin(), s.end()));
    AT_ASSERTM(heads.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(heads.dim() == 1, "0ffset tensor must be one-dimensional");
    auto heads_ptr = heads.data_ptr<NodeIDType>();
#pragma omp parallel for num_threads(threads)
    for(int64_t i=0; i<heads.size(0); i++){
        if(s.count(heads_ptr[i])==1){
        #pragma omp critical(erase)
            s.erase(heads_ptr[i]);
        }
    }
    vector<NodeIDType> ret;
    ret.reserve(s.size()+heads.numel());
    ret.assign(heads_ptr, heads_ptr+heads.numel());
    ret.insert(ret.end(), s.begin(), s.end());
    // cout<<"s: "<<s.size()<<" array: "<<array.size()<<endl;
    return th::tensor(ret);
}

int nodeIdToPartId(NodeIDType nid, const vector<NodeIDType>& part_ptr){
    int partitionId = -1;
    for(int i=0;i<part_ptr.size()-1;i++){
            if(nid>=part_ptr[i]&&nid<part_ptr[i+1]){
                partitionId = i;
                break;
            }
    }
    if(partitionId<0) throw "nid 不存在对应的分区";
    return partitionId;
}
//0:inner; 1:outer
int nodeIdToInOut(NodeIDType nid, int pid, const vector<NodeIDType>& part_ptr){
    if(nid>=part_ptr[pid]&&nid<part_ptr[pid+1]){
        return 0;
    }
    return 1;
}

vector<th::Tensor> divide_nodes_to_part(
        th::Tensor nodes, const vector<NodeIDType>& part_ptr, int threads){
    double start_time = omp_get_wtime();
    AT_ASSERTM(nodes.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(nodes.dim() == 1, "0ffset tensor must be one-dimensional");
    auto nodes_id = nodes.data_ptr<NodeIDType>();
    vector<vector<vector<NodeIDType>>> node_part_threads;
    vector<th::Tensor> result(part_ptr.size()-1);
    //初始化点的分区，每个分区按线程划分避免冲突
    for(int i = 0; i<threads; i++){
        vector<vector<NodeIDType>> node_parts;
        for(int j=0;j<part_ptr.size()-1;j++){
            node_parts.push_back(vector<NodeIDType>());
        }
        node_part_threads.push_back(node_parts);
    }
#pragma omp parallel for num_threads(threads) default(shared)
    for(int64_t i=0; i<nodes.size(0); i++){
        int tid = omp_get_thread_num();
        int pid = nodeIdToPartId(nodes_id[i], part_ptr);
        node_part_threads[tid][pid].emplace_back(nodes_id[i]);
    }
#pragma omp parallel for num_threads(part_ptr.size()-1) default(shared)
    for(int i = 0; i<part_ptr.size()-1; i++){
        vector<NodeIDType> temp;
        for(int j=0;j<threads;j++){
            temp.insert(temp.end(), node_part_threads[j][i].begin(), node_part_threads[j][i].end());
        }
        result[i]=th::tensor(temp);
    }
    double end_time = omp_get_wtime();
    // cout<<"end divide consume: "<<end_time-start_time<<"s"<<endl;
    return result;
}

float getRandomFloat(unsigned int* seed, float min, float max) {
    float scale = rand_r(seed) / (float) RAND_MAX; // 转换为0到1之间的浮点数
    return min + scale * (max - min); // 调整到min到max之间
}

NodeIDType sample_multinomial(const vector<WeightType>& weights, unsigned int* loc_seed, NodeIDType end_index){
    NodeIDType sample_indice;
    vector<WeightType> cumulative_weights;
    inclusive_scan(weights.begin(), weights.begin()+end_index, back_inserter(cumulative_weights));
    // partial_sum(weights.begin(), weights.begin()+end_index, back_inserter(cumulative_weights));
    AT_ASSERTM(cumulative_weights.back() > 0, "Edge weight sum should be greater than 0.");
    
    // uniform_real_distribution<WeightType> distribution(0.0, cumulative_weights.back());
    // WeightType random_value = distribution(e);
    WeightType random_value = getRandomFloat(loc_seed, 0.0, cumulative_weights.back());
    auto it = lower_bound(cumulative_weights.begin(), cumulative_weights.end(), random_value);
    sample_indice = distance(cumulative_weights.begin(), it);
    return sample_indice;
}

vector<int64_t> sample_max(const vector<WeightType>& weights, int k) {
    vector<int64_t> indices(weights.size());
    for (int i = 0; i < weights.size(); ++i) {
        indices[i] = i;
    }

    // 使用部分排序算法（选择算法）找到前k个最大值的索引
    partial_sort(indices.begin(), indices.begin() + k, indices.end(), 
                 [&weights](int64_t a, int64_t b) { return weights[a] > weights[b]; });

    // 返回前k个最大值的索引
    return vector<int64_t>(indices.begin(), indices.begin() + k);
}