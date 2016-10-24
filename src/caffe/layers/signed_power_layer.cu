#include <vector>

#include "caffe/layers/signed_power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    

template <typename Dtype>
void caffe_gpu_signed_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
__global__ void signed_powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
      if (a[index] >= 0) {
        y[index] = pow(a[index], alpha);
      } else {
        y[index] = -pow(-a[index], alpha);
      }
  }
}

template <>
void caffe_gpu_signed_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  signed_powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_signed_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  signed_powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}
    
template <typename Dtype>
void SignedPowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    caffe_gpu_scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_gpu_add_scalar(count, shift_, top_data);
  }
  if (power_ == Dtype(0)) {
    caffe_gpu_sign(count, top_data, top_data);
  } else if (power_ != Dtype(1)) {
    caffe_gpu_signed_powx(count, top_data, power_, top_data);
  }
}

template <typename Dtype>
void SignedPowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype eps = 0.03;
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      caffe_gpu_set(count, diff_scale_, bottom_diff);
    } else {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
        //               = diff_scale * y / (shift + scale * x)
        caffe_copy(count, bottom_data, bottom_diff);
        if (scale_ != Dtype(1)) {
          caffe_gpu_scal(count, scale_, bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          caffe_gpu_add_scalar(count, shift_, bottom_diff);
        } else {
          caffe_gpu_add_scalar(count, eps, bottom_diff);
        }
        const Dtype* top_data = top[0]->gpu_data();
        caffe_gpu_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          caffe_gpu_scal(count, diff_scale_, bottom_diff);
        }
    }
    caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SignedPowerLayer);


}  // namespace caffe
