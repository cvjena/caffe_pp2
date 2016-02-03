#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/spatial_softmax_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SpatialSoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SpatialSoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SpatialSoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SpatialSoftmaxLayerTest, TestDtypesAndDevices);

TYPED_TEST(SpatialSoftmaxLayerTest, TestForwardSpatial) {
  if (Caffe::mode() == Caffe::GPU) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    SpatialSoftmaxLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // NOTE: Written for pixel-wise softmax currently implemented in GPU, and not
    // yet in CPU.
    // Test sum
    for (int i = 0; i < this->blob_bottom_->num(); ++i) {
      for (int j = 0; j < this->blob_top_->channels(); ++j) {
        Dtype sum = 0;
        for (int k = 0; k < this->blob_bottom_->height(); ++k) {
          for (int l = 0; l < this->blob_bottom_->width(); ++l) {
            sum += this->blob_top_->data_at(i, j, k, l);
          }
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);

        // Test exact values
        Dtype scale = 0;
        for (int k = 0; k < this->blob_bottom_->height(); ++k) {
          for (int l = 0; l < this->blob_bottom_->width(); ++l) {
            scale += exp(this->blob_bottom_->data_at(i, j, k, l));
          }
        }
        for (int k = 0; k < this->blob_bottom_->height(); ++k) {
          for (int l = 0; l < this->blob_bottom_->width(); ++l) {
            EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + 1e-4,
                exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
                << "debug: " << i << " " << j;
            EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - 1e-4,
                exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
                << "debug: " << i << " " << j;
          }
        }
      }
    }
  }
}

TYPED_TEST(SpatialSoftmaxLayerTest, TestGradientSpatial) {
  if (Caffe::mode() == Caffe::GPU) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_spatial_softmax_param()->set_temperature(0.1);
    SpatialSoftmaxLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
    // lower tolerance for higher temperature
    layer_param.mutable_spatial_softmax_param()->set_temperature(1.0);
    SpatialSoftmaxLayer<Dtype> layer2(layer_param);
    GradientChecker<Dtype> checker2(1e-2, 1e-3);
    checker2.CheckGradientExhaustive(&layer2, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

const int test_shape_array[] = {2, 5, 2, 3, 2};
const vector<int> test_shape(test_shape_array, test_shape_array 
                          + sizeof(test_shape_array) / sizeof(int));

template <typename TypeParam>
class SpatialSoftmaxLayerNdTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SpatialSoftmaxLayerNdTest()
      : blob_bottom_(new Blob<Dtype>(test_shape)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SpatialSoftmaxLayerNdTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SpatialSoftmaxLayerNdTest, TestDtypesAndDevices);

TYPED_TEST(SpatialSoftmaxLayerNdTest, TestForwardSpatialNd) {
  if (Caffe::mode() == Caffe::GPU) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    SpatialSoftmaxLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // NOTE: Written for pixel-wise softmax currently implemented in GPU, and not
    // yet in CPU.
    // Test sum
    vector<int> shape = this->blob_bottom_->shape();
    vector<int> index(5);
    for (int i = 0; i < shape[0]; ++i) {
      index[0] = i;
      for (int j = 0; j < shape[1]; ++j) {
        index[1] = j;
        for (int u = 0; u < shape[2]; ++u) {
          index[2] = u;
          Dtype sum = 0;
          for (int k = 0; k < shape[3]; ++k) {
            index[3] = k;
            for (int l = 0; l < shape[4]; ++l) {
              index[4] = l;
              sum += this->blob_top_->data_at(index);
            }
          }
          EXPECT_GE(sum, 0.999);
          EXPECT_LE(sum, 1.001);

          // Test exact values
          Dtype scale = 0;
          for (int k = 0; k < shape[3]; ++k) {
            index[3] = k;
            for (int l = 0; l < shape[4]; ++l) {
              index[4] = l;
              scale += exp(this->blob_bottom_->data_at(index));
            }
          }
          for (int k = 0; k < shape[3]; ++k) {
            index[3] = k;
            for (int l = 0; l < shape[4]; ++l) {
              index[4] = l;
              EXPECT_GE(this->blob_top_->data_at(index) + 1e-4,
                  exp(this->blob_bottom_->data_at(index)) / scale)
                  << "debug: " << i << " " << j;
              EXPECT_LE(this->blob_top_->data_at(index) - 1e-4,
                  exp(this->blob_bottom_->data_at(index)) / scale)
                  << "debug: " << i << " " << j;
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(SpatialSoftmaxLayerNdTest, TestGradientSpatialNd) {
  if (Caffe::mode() == Caffe::GPU) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_spatial_softmax_param()->set_temperature(0.1);
    SpatialSoftmaxLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
    // lower tolerance for higher temperature
    layer_param.mutable_spatial_softmax_param()->set_temperature(1.0);
    SpatialSoftmaxLayer<Dtype> layer2(layer_param);
    GradientChecker<Dtype> checker2(1e-2, 1e-3);
    checker2.CheckGradientExhaustive(&layer2, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

}  // namespace caffe
