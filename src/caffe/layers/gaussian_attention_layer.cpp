#include <algorithm>
#include <vector>

#include "caffe/layers/gaussian_attention_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GaussianAttentionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  GaussianAttentionParameter param = this->layer_param_.gaussian_attention_param();
  sigma_ = param.sigma();
  if (sigma_ < 0.1)
    LOG(FATAL) << "Sigma of Gaussian attention should be larger than 0.1";
  this->blobs_.resize(1);
}

template <typename Dtype>
void GaussianAttentionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_locs_ = bottom[1]->height();
  channel_stride_ = channels_ / bottom[1]->height();
  // the mask has shape num_locs x height x width
  std::vector<int> mask_size = bottom[0]->shape();
  mask_size[1] = bottom[1]->height();
  this->blobs_[0].reset(new Blob<Dtype>(mask_size));
}

template <typename Dtype>
void GaussianAttentionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Not implemented";
}

template <typename Dtype>
void GaussianAttentionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not implemented";
}


#ifdef CPU_ONLY
STUB_GPU(GaussianAttentionLayer);
#endif

INSTANTIATE_CLASS(GaussianAttentionLayer);
REGISTER_LAYER_CLASS(GaussianAttention);
}  // namespace caffe
