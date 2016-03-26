#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/layers/python_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PythonLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PythonLayerTest()
      : blob_bottom_feats_(new Blob<Dtype>()),
        blob_bottom_locs_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // conv feature input
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    vector<int> blob_shape(4);
    blob_shape[0] = 256;
    blob_shape[1] = 1;
    blob_shape[2] = 10;
    blob_shape[3] = 2;
    this->blob_bottom_feats_->Reshape(blob_shape);
    filler.Fill(this->blob_bottom_feats_);
    blob_bottom_vec_.push_back(blob_bottom_feats_);

    // part location blob
    vector<int> blob_shape2(4);
    blob_shape2[0] = 256;
    blob_shape2[1] = 256;
    blob_shape2[2] = 6;
    blob_shape2[3] = 6;
    this->blob_bottom_locs_->Reshape(blob_shape2);
    filler.Fill(this->blob_bottom_locs_);
    blob_bottom_vec_.push_back(blob_bottom_locs_);

    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~PythonLayerTest() {
    delete blob_bottom_feats_;
    delete blob_bottom_locs_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_feats_;
  Blob<Dtype>* const blob_bottom_locs_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PythonLayerTest, TestDtypesAndDevices);

TYPED_TEST(PythonLayerTest, TestGradientExpansion) {
  typedef typename TypeParam::Dtype Dtype;
  //if (sizeof(Dtype) != 4) { return; }
  LayerParameter layer_param;
  layer_param.set_type("Python");
  //this->blob_bottom_vec_.push_back(this->blob_bottom_);
  //this->blob_top_vec_.push_back(this->blob_top_);
  layer_param.mutable_python_param()->set_module("part_layers");
  layer_param.mutable_python_param()->set_layer("CroppingLayerV2");
  //std::string param_string = "{'vocab': '/home/lisaanne/caffe-LSTM/examples/captions_add_new_word/for_debugging/mini_vocab.txt', 'batch_size': 2, 'top_names': ['expanded_labels'], 'lexical_classes': '/home/lisaanne/caffe-LSTM/examples/captions_add_new_word/for_debugging/mini_classifiers.txt'}";
  //layer_param.mutable_python_param()->set_param_str(param_string);

  shared_ptr<Layer<Dtype> > layer(LayerRegistry<Dtype>::CreateLayer(layer_param));

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
