#include <vector>

#include "caffe/layers/lrn3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	
template <typename Dtype>
void LRN3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();
  if (this->layer_param_.lrn_param().norm_region() ==
      LRNParameter_NormRegion_WITHIN_CHANNEL) {
    // Set up split_layer_ to use inputs in the numerator and denominator.
    split_top_vec_.clear();
    split_top_vec_.push_back(&product_input_);
    split_top_vec_.push_back(&square_input_);
    LayerParameter split_param;
    split_layer_.reset(new SplitLayer<Dtype>(split_param));
    split_layer_->SetUp(bottom, split_top_vec_);
    // Set up square_layer_ to square the inputs.
    square_bottom_vec_.clear();
    square_top_vec_.clear();
    square_bottom_vec_.push_back(&square_input_);
    square_top_vec_.push_back(&square_output_);
    LayerParameter square_param;
    square_param.mutable_power_param()->set_power(Dtype(2));
    square_layer_.reset(new PowerLayer<Dtype>(square_param));
    square_layer_->SetUp(square_bottom_vec_, square_top_vec_);
    // Set up pool_layer_ to sum over square neighborhoods of the input.
    pool_top_vec_.clear();
    pool_top_vec_.push_back(&pool_output_);
    LayerParameter pool_param;
    pool_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_AVE);
    pool_param.mutable_pooling_param()->set_pad(pre_pad_);
    pool_param.mutable_pooling_param()->set_kernel_size(size_);
    pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
    pool_layer_->SetUp(square_top_vec_, pool_top_vec_);
    // Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
    // the sum of a squared neighborhood (the output of pool_layer_).
    power_top_vec_.clear();
    power_top_vec_.push_back(&power_output_);
    LayerParameter power_param;
    power_param.mutable_power_param()->set_power(-beta_);
    power_param.mutable_power_param()->set_scale(alpha_);
    power_param.mutable_power_param()->set_shift(Dtype(1));
    power_layer_.reset(new PowerLayer<Dtype>(power_param));
    power_layer_->SetUp(pool_top_vec_, power_top_vec_);
    // Set up a product_layer_ to compute outputs by multiplying inputs by the
    // inverse demoninator computed by the power layer.
    product_bottom_vec_.clear();
    product_bottom_vec_.push_back(&product_input_);
    product_bottom_vec_.push_back(&power_output_);
    LayerParameter product_param;
    EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
    eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
    product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
    product_layer_->SetUp(product_bottom_vec_, top);
  }
}

template <typename Dtype>
void LRN3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(5, bottom[0]->num_axes()) << "Input must have 5 axes, "
      << "corresponding to (num, channels, length, height, width)";
  this->num_ = bottom[0]->shape(0);
  this->channels_ = bottom[0]->shape(1);
  this->length_ = bottom[0]->shape(2);
  this->height_ = bottom[0]->shape(3);
  this->width_ = bottom[0]->shape(4);
  vector<int> shape;
  for (int i=0; i<5; i++){
	shape.push_back(bottom[0]->shape(i));
  }
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    top[0]->Reshape(shape);
    this->scale_.Reshape(shape);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    this->split_layer_->Reshape(bottom, this->split_top_vec_);
    this->square_layer_->Reshape(this->square_bottom_vec_, this->square_top_vec_);
    this->pool_layer_->Reshape(this->square_top_vec_, this->pool_top_vec_);
    this->power_layer_->Reshape(this->pool_top_vec_, this->power_top_vec_);
    this->product_layer_->Reshape(this->product_bottom_vec_, top);
    break;
  }
}

template <typename Dtype>
void LRN3DLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = this->scale_.mutable_cpu_data();
  // start with the constant value
  for (int i = 0; i < this->scale_.count(); ++i) {
    scale_data[i] = this->k_;
  }
  vector<int> shape;
  shape.push_back(1);
  shape.push_back(this->channels_ + this->size_ - 1);
  shape.push_back(this->length_);
  shape.push_back(this->height_);
  shape.push_back(this->width_);
  Blob<Dtype> padded_square(shape);
  Dtype* padded_square_data = padded_square.mutable_cpu_data();
  caffe_set(padded_square.count(), Dtype(0), padded_square_data);
  Dtype alpha_over_size = this->alpha_ / this->size_;
  // go through the images
  for (int n = 0; n < this->num_; ++n) {
    // compute the padded square

    caffe_sqr(this->channels_ * this->length_ * this->height_ * this->width_,
        bottom_data + bottom[0]->offset(n ,0,0,0,0),
        padded_square_data + padded_square.offset(0, this->pre_pad_ ,0,0,0));
		//LOG(INFO) << "debug";
    // Create the first channel scale
    for (int c = 0; c < this->size_; ++c) {
      caffe_axpy<Dtype>(this->length_ * this->height_ * this->width_, alpha_over_size,
          padded_square_data + padded_square.offset(0, c ,0,0,0),
          scale_data + this->scale_.offset(n, 0 ,0,0,0));
    }
    for (int c = 1; c < this->channels_; ++c) {
      // copy previous scale
      caffe_copy<Dtype>(this->length_ * this->height_ * this->width_,
          scale_data + this->scale_.offset(n, c - 1 ,0,0,0),
          scale_data + this->scale_.offset(n, c ,0,0,0));
      // add head
      caffe_axpy<Dtype>(this->length_ * this->height_ * this->width_, alpha_over_size,
          padded_square_data + padded_square.offset(0, c + this->size_ - 1 ,0,0,0),
          scale_data + this->scale_.offset(n, c ,0,0,0));
      // subtract tail
      caffe_axpy<Dtype>(this->length_ * this->height_ * this->width_, -alpha_over_size,
          padded_square_data + padded_square.offset(0, c - 1 ,0,0,0),
          scale_data + this->scale_.offset(n, c ,0,0,0));
    }
  }

  // In the end, compute output
  caffe_powx<Dtype>(this->scale_.count(), scale_data, -this->beta_, top_data);
  caffe_mul<Dtype>(this->scale_.count(), top_data, bottom_data, top_data);
}

template <typename Dtype>
void LRN3DLayer<Dtype>::WithinChannelForward(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(bottom, split_top_vec_);
  square_layer_->Forward(square_bottom_vec_, square_top_vec_);
  pool_layer_->Forward(square_top_vec_, pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, top);
}

template <typename Dtype>
void LRN3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_cpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRN3DLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = this->scale_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  vector<int> shape;
  shape.push_back(1);
  shape.push_back(this->channels_ + this->size_ - 1);
  shape.push_back(this->length_);
  shape.push_back(this->height_);
  shape.push_back(this->width_);
  Blob<Dtype> padded_ratio(shape);
  shape[1] = 1;
  Blob<Dtype> accum_ratio(shape);
  Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
  Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
  // We hack a little bit by using the diff() to store an additional result
  Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
  caffe_set(padded_ratio.count(), Dtype(0), padded_ratio_data);
  Dtype cache_ratio_value = 2. * this->alpha_ * this->beta_ / this->size_;

  caffe_powx<Dtype>(this->scale_.count(), scale_data, -this->beta_, bottom_diff);
  caffe_mul<Dtype>(this->scale_.count(), top_diff, bottom_diff, bottom_diff);

  // go through individual data
  int inverse_pre_pad = this->size_ - (this->size_ + 1) / 2;
  for (int n = 0; n < this->num_; ++n) {
    int block_offset = this->scale_.offset(n ,0,0,0,0);
    // first, compute diff_i * y_i / s_i
    caffe_mul<Dtype>(this->channels_ * this->length_ * this->height_ * this->width_,
        top_diff + block_offset, top_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad ,0,0,0));
    caffe_div<Dtype>(this->channels_ * this->length_ * this->height_ * this->width_,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad ,0,0,0),
        scale_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad ,0,0,0));
    // Now, compute the accumulated ratios and the bottom diff
    caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
    for (int c = 0; c < this->size_ - 1; ++c) {
      caffe_axpy<Dtype>(this->length_ * this->height_ * this->width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c ,0,0,0), accum_ratio_data);
    }
    for (int c = 0; c < this->channels_; ++c) {
      caffe_axpy<Dtype>(this->length_ * this->height_ * this->width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c + this->size_ - 1 ,0,0,0),
          accum_ratio_data);
      // compute bottom diff
      caffe_mul<Dtype>(this->length_ * this->height_ * this->width_,
          bottom_data + top[0]->offset(n, c ,0,0,0),
          accum_ratio_data, accum_ratio_times_bottom);
      caffe_axpy<Dtype>(this->length_ * this->height_ * this->width_, -cache_ratio_value,
          accum_ratio_times_bottom, bottom_diff + top[0]->offset(n, c ,0,0,0));
      caffe_axpy<Dtype>(this->length_ * this->height_ * this->width_, -1.,
          padded_ratio_data + padded_ratio.offset(0, c ,0,0,0), accum_ratio_data);
    }
  }
}

template <typename Dtype>
void LRN3DLayer<Dtype>::WithinChannelBackward(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    vector<bool> product_propagate_down(2, true);
    product_layer_->Backward(top, product_propagate_down, product_bottom_vec_);
    power_layer_->Backward(power_top_vec_, propagate_down, pool_top_vec_);
    pool_layer_->Backward(pool_top_vec_, propagate_down, square_top_vec_);
    square_layer_->Backward(square_top_vec_, propagate_down,
                            square_bottom_vec_);
    split_layer_->Backward(split_top_vec_, propagate_down, bottom);
  }
}

template <typename Dtype>
void LRN3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_cpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(LRN3DLayer);
STUB_GPU_FORWARD(LRN3DLayer, CrossChannelForward);
STUB_GPU_BACKWARD(LRN3DLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(LRN3DLayer);
REGISTER_LAYER_CLASS(LRN3D);

}  // namespace caffe
