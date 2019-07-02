/*
 *
 *  Copyright (c) 2015, Facebook, Inc. All rights reserved.
 *
 *  Licensed under the Creative Commons Attribution-NonCommercial 3.0
 *  License (the "License"). You may obtain a copy of the License at
 *  https://creativecommons.org/licenses/by-nc/3.0/.
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *
 */

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <stdio.h>

#include "caffe/common.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;


namespace caffe {


void ImageToBuffer(const cv::Mat* img, char* buffer){
	int idx = 0;
	for (int c = 0; c < 3; ++c) {
	  for (int h = 0; h < img->rows; ++h) {
		for (int w = 0; w < img->cols; ++w) {
			buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
		}
	  }
	}
}
void ImageChannelToBuffer(const cv::Mat* img, char* buffer, int c){
    int idx = 0;
	for (int h = 0; h < img->rows; ++h) {
	    for (int w = 0; w < img->cols; ++w) {
		    buffer[idx++] = img->at<cv::Vec3b>(h, w)[c];
		}
	}
}

void GrayImageToBuffer(const cv::Mat* img, char* buffer){
	int idx = 0;
    for (int h = 0; h < img->rows; ++h) {
	  for (int w = 0; w < img->cols; ++w) {
		buffer[idx++] = img->at<unsigned char>(h, w);
	  }
	}
}
void BufferToGrayImage(const char* buffer, const int height, const int width, cv::Mat* img){
	int idx = 0;
	img->create(height, width, CV_8U);
    for (int h = 0; h < height; ++h) {
	  for (int w = 0; w < width; ++w) {
		img->at<unsigned char>(h, w) = buffer[idx++];
	  }
	}
}
void BufferToGrayImage(const float* buffer, const int height, const int width, cv::Mat* img){
	int idx = 0;
	img->create(height, width, CV_8U);
    for (int h = 0; h < height; ++h) {
	  for (int w = 0; w < width; ++w) {
		img->at<unsigned char>(h, w) = (unsigned char)(buffer[idx++]);
	  }
	}
}
void BufferToGrayImage(const double* buffer, const int height, const int width, cv::Mat* img){
	int idx = 0;
	img->create(height, width, CV_8U);
    for (int h = 0; h < height; ++h) {
	  for (int w = 0; w < width; ++w) {
		img->at<unsigned char>(h, w) = (unsigned char)(buffer[idx++]);
	  }
	}
}

void BufferToColorImage(const char* buffer, const int height, const int width, cv::Mat* img){
	img->create(height, width, CV_8UC3);
	for (int c=0; c<3; c++) {
		for (int h = 0; h < height; ++h) {
		  for (int w = 0; w < width; ++w) {
			img->at<cv::Vec3b>(h, w)[c] = buffer[c * width * height + h * width + w];
		  }
		}
	}
}

bool ReadVideoToVolumeDatum(const char* filename, const int start_frm,
  const int label, const int length, const int height, const int width,
  const int sampling_rate, VolumeDatum* datum){
  if (!ReadVideoToVolumeDatumHelper(filename, start_frm, label, length,
      height, width, sampling_rate, datum)) {
      return ReadVideoToVolumeDatumHelperSafe(filename, start_frm, label,
        length, height, width, sampling_rate, datum);
  } else
      return true;
}

bool ReadVideoToVolumeDatumHelper(const char* filename, const int start_frm,
  const int label, const int length, const int height, const int width,
  const int sampling_rate, VolumeDatum* datum){
  cv::VideoCapture cap;
  cv::Mat img, img_origin;
  char *buffer = NULL;
  int offset = 0;
 int channel_size = 0;
 int image_size = 0;
 int data_size = 0;
  int use_start_frm = start_frm;

  cap.open(filename);
  if (!cap.isOpened()){
    LOG(ERROR) << "Cannot open " << filename;
    return false;
  }

  datum->set_channels(3);
  datum->set_length(length);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();

  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  if (num_of_frames<length*sampling_rate){
    LOG(INFO) << filename << " does not have enough frames; having "
    << num_of_frames;
    return false;
  }

  // If start_frame == -1, then use random temporal jitering
  if (start_frm < 0){
    use_start_frm = caffe_rng_rand()%(num_of_frames-length*sampling_rate+1);
  }

  offset = 0;
  CHECK_GE(use_start_frm, 0) << "start frame must be greater or equal to 0";

  if (use_start_frm)
    cap.set(CV_CAP_PROP_POS_FRAMES, use_start_frm);

  int end_frm = use_start_frm + length * sampling_rate;
  CHECK_LE(end_frm, num_of_frames)
    << "end frame must be less or equal to num of frames";

  for (int i=use_start_frm; i<end_frm; i+=sampling_rate){
		if (sampling_rate > 1)
			cap.set(CV_CAP_PROP_POS_FRAMES, i);
		if (height > 0 && width > 0){
			cap.read(img_origin);
			if (!img_origin.data){
				LOG(INFO) << filename << " has no data at frame " << i;
				if (buffer!=NULL)
					delete[] buffer;
				return false;
			}
			cv::resize(img_origin, img, cv::Size(width, height));
		}
		else
			cap.read(img);
		if (!img.data){
			LOG(ERROR) << "Could not open or find file " << filename;
			if (buffer!=NULL)
				delete[] buffer;
			return false;
		}

		if (i==use_start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length;
			data_size = channel_size * 3;
			buffer = new char[data_size];
		}
		for (int c=0; c<3; c++){
			ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
		}
		offset += image_size;
	}
	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
	cap.release();
 	return true;
}

bool ReadVideoToVolumeDatumHelperSafe(const char* filename, const int start_frm,
  const int label, const int length, const int height, const int width,
  const int sampling_rate, VolumeDatum* datum){
  cv::VideoCapture cap;
	cv::Mat img, img_origin;
  char *buffer = NULL;
 int offset = 0;
 int channel_size = 0;
 int image_size = 0;
 int data_size = 0;
	int use_start_frm = start_frm;

	cap.open(filename);
	if (!cap.isOpened()){
		LOG(ERROR) << "Cannot open " << filename;
		return false;
	}

	datum->set_channels(3);
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	if (num_of_frames<length*sampling_rate){
		LOG(INFO) << filename << " does not have enough frames; having "
    << num_of_frames;
		return false;
	}
	if (start_frm < 0){
		use_start_frm = caffe_rng_rand()%(num_of_frames-length*sampling_rate+1);
	}

	offset = 0;
	CHECK_GE(use_start_frm, 0) << "start frame must be greater or equal to 0";

  // Instead of random acess, do sequentically access (avoid key-frame issue)
	// This will keep use_start_frm frames
	int sequential_counter = 0;
	while (sequential_counter < use_start_frm) {
		cap.read(img_origin);
		sequential_counter++;
	}

	int end_frm = use_start_frm + length * sampling_rate;
	CHECK_LE(end_frm, num_of_frames) << "end frame must be less or equal to num of frames";

	for (int i=use_start_frm; i<end_frm; i++){
		if (sampling_rate > 1) {
			// If sampling_rate > 1, purposely keep some frames
			if ((i-use_start_frm) % sampling_rate !=0) {
				cap.read(img_origin);
				continue;
			}
		}
		if (height > 0 && width > 0){
			cap.read(img_origin);
			if (!img_origin.data){
				LOG(INFO) << filename << " has no data at frame " << i;
				if (buffer!=NULL)
					delete[] buffer;
				return false;
			}
			cv::resize(img_origin, img, cv::Size(width, height));
		}
		else
			cap.read(img);
		if (!img.data){
			LOG(ERROR) << "Could not open or find file " << filename;
			if (buffer!=NULL)
				delete[] buffer;
			return false;
		}

		if (i==use_start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length;
			data_size = channel_size * 3;
			buffer = new char[data_size];
		}
		for (int c=0; c<3; c++){
			ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
		}
		offset += image_size;
	}
	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
	cap.release();
	return true;
}


bool ReadImageSequenceToVolumeDatum(const char* img_dir, const int start_frm, const int label,
		const int length, const int height, const int width, const int sampling_rate, VolumeDatum* datum){
	char fn_im[256];
	cv::Mat img, img_origin;
	char *buffer = NULL;
 int offset = 0;
 int channel_size = 0;
 int image_size = 0;
 int data_size = 0;

	datum->set_channels(3);
	datum->set_length(length);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	offset = 0;
	int end_frm = start_frm + length * sampling_rate;
	for (int i=start_frm; i<end_frm; i+=sampling_rate){
		sprintf(fn_im, "%s/%06d.jpg", img_dir, i);
		if (height > 0 && width > 0) {
		    img_origin = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
		    if (!img_origin.data) {
			LOG(ERROR) << "Could not open or find file " << fn_im;
		    	return false;
		    }
		    cv::resize(img_origin, img, cv::Size(width, height));
		    img_origin.release();
		} else {
		  img = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
		}

		if (!img.data){
			LOG(ERROR) << "Could not open or find file " << fn_im;
			return false;
		}

		if (i==start_frm){
			datum->set_height(img.rows);
			datum->set_width(img.cols);
			image_size = img.rows * img.cols;
			channel_size = image_size * length;
			data_size = channel_size * 3;
			buffer = new char[data_size];
		}
		for (int c=0; c<3; c++){
			ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
		}
		offset += image_size;
	}
	CHECK(offset == channel_size) << "wrong offset size" << std::endl;
	datum->set_data(buffer, data_size);
	delete []buffer;
 	return true;
}

void BufferToMatrix(const float* buffer, const int height, const int width, cv::Mat* img){
	int idx = 0;
	img->create(height, width, CV_32F);
    for (int h = 0; h < height; ++h) {
	  for (int w = 0; w < width; ++w) {
		img->at<float>(h, w) = buffer[idx++];
	  }
	}
}

void MatrixToBuffer(const cv::Mat* img, float* buffer){
	int idx = 0;
    for (int h = 0; h < img->rows; ++h) {
	  for (int w = 0; w < img->cols; ++w) {
		buffer[idx++] = img->at<float>(h, w);
	  }
	}
}

bool load_volume_datum_from_binary(const string fn_blob, const int label, const int height, const int width, VolumeDatum* datum){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	
	int n, c, l, w, h;
	float* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);

	n = 1;
	int data_size = n * c * l * h * w;
	fread(buff, sizeof(float), data_size, f);
	fclose(f);
	
	//=====================
	
	datum->set_channels(c);
	datum->set_length(l);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();
	
	if (height > 0 && width > 0) {
		//=====================
		// resize
		datum->set_height(height);
		datum->set_width(width);
		int image_size = height * width;
		int channel_size = image_size * l;
		data_size = channel_size * c;
		float* buffer = new float[data_size];
		
		int offset = 0;
		int offset_im_size = 0;
		cv::Mat img;
		float* tempBuff;
		for (int im_idx=0; im_idx<l; im_idx++){
			for (int c_idx=0; c_idx<c; c_idx++){
				tempBuff = buff + c_idx * (h*w*l) + offset_im_size;
				BufferToMatrix(tempBuff, h, w, &img);
				cv::resize(img, img, cv::Size(width, height));
				MatrixToBuffer(&img, buffer + c_idx * channel_size + offset);
				img.release();
			}
			offset += image_size;
			offset_im_size += h * w;
		}
		
		for (int i=0; i<data_size; i++){
			datum->add_float_data(buffer[i]);
		}
		
		delete[] buffer;
	} else {
		//=====================
		// not resize
		datum->set_height(h);
		datum->set_width(w);
		for (int i=0; i<data_size; i++){
			datum->add_float_data(buff[i]);
		}
	}
	delete []buff;
	
	return true;
}

bool load_volume_datum_from_binary(const string fn_blob, const int label, const int height, const int width, const int length, VolumeDatum* datum){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	
	int n, c, l, w, h;
	
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	
	//LOG(INFO) << fn_blob << std::endl;
	//LOG(INFO) << n << " " << c << " " << l << " " << h << " " << w << std::endl;
	
	if(length > l){
		return false;
	}
	
	n = 1;
	int data_size = n * c * l * h * w;
	float* buff = new float[data_size];
	fread(buff, sizeof(float), data_size, f);
	
	fclose(f);
	
	//=====================
	
	datum->set_channels(c);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();
	
	if(length <= 0)
		datum->set_length(l);
	else
		datum->set_length(length);
	
	if (height > 0 && width > 0) {
		//=====================
		// resize
		datum->set_height(height);
		datum->set_width(width);
		
		int l_n = length > l ? l : length;
		
		int image_size = height * width;
		int channel_size = image_size * l_n;
		data_size = channel_size * c;
		float* buffer = new float[data_size];
		
		int offset = 0;
		int offset_im_size = 0;
		cv::Mat img;
		float* tempBuff;
		for (int im_idx=0; im_idx<l_n; im_idx++){
			for (int c_idx=0; c_idx<c; c_idx++){
				tempBuff = buff + c_idx * (h*w*l) + offset_im_size;
				BufferToMatrix(tempBuff, h, w, &img);
				cv::resize(img, img, cv::Size(width, height));
				MatrixToBuffer(&img, buffer + c_idx * channel_size + offset);
				img.release();
			}
			offset += image_size;
			offset_im_size += h * w;
		}
		
		for (int i=0; i<data_size; i++){
			datum->add_float_data(buffer[i]);
		}
		
		delete[] buffer;
	} else {
		//=====================
		// not resize
		datum->set_height(h);
		datum->set_width(w);
		
		if(length <= 0) {
			for (int i=0; i<data_size; i++){
				datum->add_float_data(buff[i]);
			}
		} else {
			int l_n = length > l ? l : length;
		
			int image_size = h * w;
			data_size = image_size * l_n * c;
			float* buffer = new float[data_size];

			int offset = 0;
			int offset_im_size = 0;
			float* tempBuff;
			for (int im_idx=0; im_idx<l_n; im_idx++){
				for (int c_idx=0; c_idx<c; c_idx++){
					tempBuff = buff + c_idx * (image_size * l) + offset_im_size;
					for (int i=0; i<image_size; i++){
						(buffer + c_idx * (image_size * l_n) + offset)[i] = tempBuff[i];
					}
				}
				offset += image_size;
				offset_im_size += image_size;
			}

			for (int i=0; i<data_size; i++){
				datum->add_float_data(buffer[i]);
			}

			delete[] buffer;
		}
	}
	delete[] buff;
	
	return true;
}

bool load_volume_datum_from_binary(const string fn_blob, const int label, const int height, const int width, const int length, const int border_line, VolumeDatum* datum){

	if(border_line <= 0)
	{
		return load_volume_datum_from_binary(fn_blob, label, height, width, length, border_line, datum);
	}

	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	
	int n, c, l, w, h;
	
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	
	//LOG(INFO) << fn_blob << std::endl;
	//LOG(INFO) << n << " " << c << " " << l << " " << h << " " << w << std::endl;
	
	if(length > l){
		return false;
	}
	
	n = 1;
	int data_size = n * c * l * h * w;
	float* buff = new float[data_size];
	fread(buff, sizeof(float), data_size, f);
	
	fclose(f);
	
	//=====================
	
	datum->set_channels(c);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();
	
	if(length <= 0)
		datum->set_length(l);
	else
		datum->set_length(length);
	
	if (height > 0 && width > 0) {
		//=====================
		// resize
		datum->set_height(height);
		datum->set_width(width);
		
		int l_n = length > l ? l : length;
		
		int image_size = height * width;
		int channel_size = image_size * l_n;
		data_size = channel_size * c;
		float* buffer = new float[data_size];
		
		int offset = 0;
		int offset_im_size = 0;
		cv::Mat img;
		cv::Mat croppedImg;
		float* tempBuff;
		for (int im_idx=0; im_idx<l_n; im_idx++){
			for (int c_idx=0; c_idx<c; c_idx++){
				tempBuff = buff + c_idx * (h*w*l) + offset_im_size;
				BufferToMatrix(tempBuff, h, w, &img);

				// remove border line
				img(cv::Rect(border_line,border_line,img.cols-2*border_line,img.rows-2*border_line)).copyTo(croppedImg);
				img = croppedImg;
				croppedImg.release();

				cv::resize(img, img, cv::Size(width, height));
				MatrixToBuffer(&img, buffer + c_idx * channel_size + offset);
				img.release();
			}
			offset += image_size;
			offset_im_size += h * w;
		}
		
		for (int i=0; i<data_size; i++){
			datum->add_float_data(buffer[i]);
		}
		
		delete[] buffer;
	} else {
		//=====================
		// not resize
		datum->set_height(h);
		datum->set_width(w);
		
		if(length <= 0) {
			for (int i=0; i<data_size; i++){
				datum->add_float_data(buff[i]);
			}
		} else {
			int l_n = length > l ? l : length;
		
			int image_size = h * w;
			int channel_size = image_size * l_n;
			data_size = image_size * l_n * c;
			float* buffer = new float[data_size];

			int offset = 0;
			int offset_im_size = 0;
			float* tempBuff;
			cv::Mat img;
			cv::Mat croppedImg;
			for (int im_idx=0; im_idx<l_n; im_idx++){
				for (int c_idx=0; c_idx<c; c_idx++){
					tempBuff = buff + c_idx * (image_size * l) + offset_im_size;

					BufferToMatrix(tempBuff, h, w, &img);
					// remove border line
					img(cv::Rect(border_line,border_line,img.cols-2*border_line,img.rows-2*border_line)).copyTo(croppedImg);
					img = croppedImg;
					croppedImg.release();

					MatrixToBuffer(&img, buffer + c_idx * channel_size + offset);
					img.release();
					
					//for (int i=0; i<image_size; i++){
					//	(buffer + c_idx * (image_size * l_n) + offset)[i] = tempBuff[i];
					//}
				}
				offset += image_size;
				offset_im_size += image_size;
			}

			for (int i=0; i<data_size; i++){
				datum->add_float_data(buffer[i]);
			}

			delete[] buffer;
		}
	}
	delete[] buff;
	
	return true;
}
	
template <>
bool load_blob_from_binary<float>(const string fn_blob, Blob<float>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	float* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	vector<int> shape(5);
	shape[0] = n;
	shape[1] = c;
	shape[2] = l;
	shape[3] = h;
	shape[4] = w;

	blob->Reshape(shape);
	buff = blob->mutable_cpu_data();

	fread(buff, sizeof(float), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool load_blob_from_binary<double>(const string fn_blob, Blob<double>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	double* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);
	vector<int> shape(5);
	shape[0] = n;
	shape[1] = c;
	shape[2] = l;
	shape[3] = h;
	shape[4] = w;

	blob->Reshape(shape);
	buff = blob->mutable_cpu_data();

	fread(buff, sizeof(double), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool load_blob_from_uint8_binary<float>(const string fn_blob, Blob<float>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	float* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);

	vector<int> shape(5);
	shape[0] = n;
	shape[1] = c;
	shape[2] = l;
	shape[3] = h;
	shape[4] = w;
	blob->Reshape(shape);

	buff = blob->mutable_cpu_data();

	int count = n * c * l * h * w;
	unsigned char* temp_buff = new unsigned char[count];

	fread(temp_buff, sizeof(unsigned char), count, f);
	fclose(f);

	for (int i = 0; i < count; i++)
		buff[i] = (float)temp_buff[i];

	delete []temp_buff;
	return true;
}

template <>
bool load_blob_from_uint8_binary<double>(const string fn_blob, Blob<double>* blob){
	FILE *f;
	f = fopen(fn_blob.c_str(), "rb");
	if (f==NULL)
		return false;
	int n, c, l, w, h;
	double* buff;
	fread(&n, sizeof(int), 1, f);
	fread(&c, sizeof(int), 1, f);
	fread(&l, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	fread(&w, sizeof(int), 1, f);

	vector<int> shape(5);
	shape[0] = n;
	shape[1] = c;
	shape[2] = l;
	shape[3] = h;
	shape[4] = w;
	blob->Reshape(shape);

	buff = blob->mutable_cpu_data();


	int count = n * c * l * h * w;
	unsigned char* temp_buff = new unsigned char[count];

	fread(temp_buff, sizeof(unsigned char), count, f);
	fclose(f);

	for (int i = 0; i < count; i++)
		buff[i] = (double)temp_buff[i];

	delete []temp_buff;
	return true;
}


template <>
bool save_blob_to_binary<float>(Blob<float>* blob, const string fn_blob, int num_index){
	FILE *f;
	float *buff;
	int n, c, l, w, h;
	f = fopen(fn_blob.c_str(), "wb");
	if (f==NULL)
		return false;

	c = blob->shape(1);
	if (blob->num_axes() > 2)
		l = blob->shape(2);
	else
		l = 1;
	if (blob->num_axes() > 3)
		h = blob->shape(3);
	else
		h = 1;
	if (blob->num_axes() > 4)
		w = blob->shape(4);
	else
		w = 1;

	if (num_index<0){
		n = blob->shape(0);
		buff = blob->mutable_cpu_data();
	}else{
		n = 1;
		buff = blob->mutable_cpu_data() + num_index * c * l * h * w;
	}

	fwrite(&n, sizeof(int), 1, f);
	fwrite(&c, sizeof(int), 1, f);
	fwrite(&l, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(&w, sizeof(int), 1, f);
	fwrite(buff, sizeof(float), n * c * l * h * w, f);
	fclose(f);
	return true;
}

template <>
bool save_blob_to_binary<double>(Blob<double>* blob, const string fn_blob, int num_index){
	FILE *f;
	double *buff;
	int n, c, l, w, h;
	f = fopen(fn_blob.c_str(), "wb");
	if (f==NULL)
		return false;

	c = blob->shape(1);
	if (blob->num_axes() > 2)
		l = blob->shape(2);
	else
		l = 1;
	if (blob->num_axes() > 3)
		h = blob->shape(3);
	else
		h = 1;
	if (blob->num_axes() > 4)
		w = blob->shape(4);
	else
		w = 1;
	if (num_index<0){
		n = blob->shape(0);
		buff = blob->mutable_cpu_data();
	}else{
		n = 1;
		buff = blob->mutable_cpu_data() + num_index * c * l * h * w;
	}


	fwrite(&n, sizeof(int), 1, f);
	fwrite(&c, sizeof(int), 1, f);
	fwrite(&l, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(&w, sizeof(int), 1, f);
	fwrite(buff, sizeof(double), n * c * l * h * w, f);
	fclose(f);
	return true;
}


}
