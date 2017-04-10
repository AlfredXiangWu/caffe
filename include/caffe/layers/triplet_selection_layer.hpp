#ifndef CAFFE_TRIPLET_SELECTION_LAYER_HPP_
#define CAFFE_TRIPLET_SELECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class TripletSelectionLayer : public Layer<Dtype> {
 public:
  explicit TripletSelectionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TripletSelection"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 3; }
  inline Dtype compute_L2_distance(const Dtype* feat1, const Dtype* feat2, int feat_len) {
      Dtype distance = 0.0;
      for(int i = 0; i < feat_len; ++i) {
          distance += (feat1[i] - feat2[i])*(feat1[i] - feat2[i]);
      }
      return distance;
  }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  float thr_;
  int max_num_;

  vector<int> idx_anc, idx_pos, idx_neg;

};

} // namespace caffe

#endif