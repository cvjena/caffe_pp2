#include <algorithm>
#include <vector>

#include "caffe/layers/gaussian_attention_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void CreateMask(const int nthreads,
    const Dtype* const attention_locs,
    const int num, const int num_locs,
    const int height, const int width, 
    const Dtype sigma, 
    Dtype* const mask) {
  // For each mask entry
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int l = (index / width / height) % num_locs;
    const int n = index / width / height / num_locs;
    // dim=0 is along width and dim=1 along height, with -inf towards the top left
    Dtype cur_attention[2] = { attention_locs[n*(2*num_locs) + 2*l], attention_locs[n*(2*num_locs) +  2*l+1]};
    Dtype cur_loc[2] = { Dtype(w) / Dtype(width-1), Dtype(h) / Dtype(height-1)};
    
    mask[index] = Dtype(1.0);
    Dtype y;
    for (int dim = 0; dim < 2; dim++)
    {
      // Make cure everything is within valid ranged
      cur_attention[dim] = min(Dtype(1.0),max(Dtype(-1.0),cur_attention[dim]));
      // Transform the current location to be in [-1,1]
      cur_loc[dim] = cur_loc[dim]*Dtype(2.0)-Dtype(1.0);
      // The input location for scipy.stats.norm.pdf(x,loc=loc,scale=sigma)
      y = (cur_loc[dim] - cur_attention[dim]) / sigma;
      mask[index] = mask[index] * ( exp(-y * y / Dtype(2.0) ) / sigma / sqrt(Dtype(2.0 * 3.141592653589793)) );
    }
  }
}    
  
  
template <typename Dtype>
__global__ void MultiplyMask(const int nthreads,
    const Dtype* const bottom_data,
    const Dtype* const mask,
    const int channels,
    const int height, const int width, 
    const int num_locs, const int channel_stride,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    
    const int mask_id = c / channel_stride;
    const int mask_index = n*num_locs*width*height + mask_id*width*height + (index%(width*height));
    
    top_data[index] = bottom_data[index] * mask[mask_index];
  }
}  
  
  
  
template <typename Dtype>
void GaussianAttentionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* attention_locs = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  Dtype* mask_data = this->blobs_[0]->mutable_gpu_data();
  
  // NOLINT_NEXT_LINE(whitespace/operators)
  CreateMask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      this->blobs_[0]->count(), attention_locs,
      this->blobs_[0]->num(), num_locs_,
      height_, width_, 
      sigma_, mask_data);
  
  const Dtype* const_mask_data = this->blobs_[0]->gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  MultiplyMask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, mask_data,
      channels_,
      height_, width_, 
      // the number of locs
      num_locs_, channel_stride_,
      top_data);
  
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
void GaussianAttentionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not implemented";
}

INSTANTIATE_LAYER_GPU_FUNCS(GaussianAttentionLayer);


}  // namespace caffe
