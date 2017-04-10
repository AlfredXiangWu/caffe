#include <algorithm>
#include <vector>

#include "caffe/layers/triplet_selection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletSelectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    max_num_ = this->layer_param_.triplet_selection_param().max_num();
    thr_ = this->layer_param_.triplet_selection_param().thr();
}


template <typename Dtype>
void TripletSelectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)  {
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;

    vector<int> top_shape(4);
    top_shape[0] = max_num_;
    top_shape[1] = dim;
    top_shape[3] = bottom[0]->width();
    top_shape[4] = bottom[0]->height();

    top[0]->Reshape(top_shape);   // anchor
    top[1]->Reshape(top_shape);   // pos
    top[2]->Reshape(top_shape);   // neg
}

template <typename Dtype>
void TripletSelectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;

    const Dtype* feat  = bottom[0]->cpu_data(); 
    const Dtype* label = bottom[1]->cpu_data();

    // Triplet selection
    int cnt = 0;
    // anchor
    for(int i = 0; i < num; ++i) {
        const Dtype* feat_anchor = feat + (i*dim);
        int label_anchor   = static_cast<int>(label[i]);
        // pos
        for(int j = i + 1; j < num; ++j) {
            const Dtype* feat_pos = feat + (j*dim);
            int label_pos   = static_cast<int>(label[j]);
            if(label_anchor != label_pos)
                continue;
            Dtype score_pos = compute_L2_distance(feat_anchor, feat_pos, dim);
            // neg
            while(cnt < max_num_) {
                int k = rand() % num;
                const Dtype* feat_neg = feat + (k*dim);
                int label_neg   = static_cast<int>(label[k]);
                if(label_anchor == label_neg)
                    continue;
                Dtype score_neg = compute_L2_distance(feat_anchor, feat_neg, dim);
                if((score_pos + thr_ > score_neg) && (score_pos < score_neg)) {
                    idx_anc.push_back(i);
                    idx_pos.push_back(j);
                    idx_neg.push_back(k);
                    cnt++;
                }
            } // for(int k =0; k < num; ++k)
        } //for(int j = i + 1; j < num; ++j) 
    } // for(int i = 0; i < num; ++i)

    for(int i = 0; i < idx_anc.size(); ++i) {
        caffe_copy<Dtype>(dim, feat+idx_anc[i]*dim, top[0]->mutable_cpu_data()+i*dim);
        caffe_copy<Dtype>(dim, feat+idx_pos[i]*dim, top[1]->mutable_cpu_data()+i*dim);
        caffe_copy<Dtype>(dim, feat+idx_neg[i]*dim, top[2]->mutable_cpu_data()+i*dim);
    }
}

template <typename Dtype>
void TripletSelectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;

    const Dtype* diff_anc = top[0]->cpu_diff();
    const Dtype* diff_pos = top[1]->cpu_diff();
    const Dtype* diff_neg = top[2]->cpu_diff();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    int cnt[num];
    caffe_set(num, int(0), cnt);
    caffe_set(num*dim, Dtype(0), bottom_diff);

    for(int i = 0; i < idx_anc.size(); ++i) {
        caffe_add(dim, diff_anc+dim*i, bottom[0]->cpu_diff(), bottom_diff+dim*idx_anc[i]);
        cnt[idx_anc[i]]++;
        caffe_add(dim, diff_pos+dim*i, bottom[0]->cpu_diff(), bottom_diff+dim*idx_pos[i]);
        cnt[idx_pos[i]]++;
        caffe_add(dim, diff_neg+dim*i, bottom[0]->cpu_diff(), bottom_diff+dim*idx_neg[i]);
        cnt[idx_neg[i]]++;
    }

    for(int i = 0; i < num; ++i) {
        if(cnt[i] != 0) {
            caffe_scal<Dtype>(dim, 1 / Dtype(cnt[i]), bottom_diff+dim*i);
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(TripletSelectionLayer);
#endif

INSTANTIATE_CLASS(TripletSelectionLayer);
REGISTER_LAYER_CLASS(TripletSelection);

} // namespace caffe