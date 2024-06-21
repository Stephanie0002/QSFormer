#pragma once
#include "head.h"

struct TemporalGraphBlock
{
    public:
        vector<NodeIDType> row;
        vector<NodeIDType> col;
        th::Tensor eid;
        vector<TimeStampType> delta_ts;
        th::Tensor src_index;
        th::Tensor sample_nodes;
        th::Tensor sample_nodes_ts;
        vector<WeightType> e_weights;
        double sample_time = 0;
        double tot_time = 0;
        double extract_time = 0;
        double compute_time = 0;
        int64_t sample_edge_num = 0;

        TemporalGraphBlock(){}
        // TemporalGraphBlock(const TemporalGraphBlock &tgb);
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           th::Tensor &_sample_nodes):
                           row(_row), col(_col), sample_nodes(_sample_nodes){}
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           th::Tensor &_sample_nodes,
                           th::Tensor &_sample_nodes_ts):
                           row(_row), col(_col), sample_nodes(_sample_nodes),
                           sample_nodes_ts(_sample_nodes_ts){}
};

class T_TemporalGraphBlock
{
    public:
        th::Tensor row;
        th::Tensor col;
        th::Tensor eid;
        th::Tensor delta_ts;
        th::Tensor src_index;
        th::Tensor sample_nodes;
        th::Tensor sample_nodes_ts;
        th::Tensor e_weights;
        double sample_time = 0;
        double tot_time = 0;
        double extract_time = 0;
        double compute_time = 0;
        int64_t sample_edge_num = 0;

        T_TemporalGraphBlock(){}
        T_TemporalGraphBlock(th::Tensor &_row, th::Tensor &_col,
                           th::Tensor &_sample_nodes):
                           row(_row), col(_col), sample_nodes(_sample_nodes){}
};