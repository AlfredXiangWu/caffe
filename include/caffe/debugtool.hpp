#ifndef CAFFE_DEBUGTOOL_H_
#define CAFFE_DEBUGTOOL_H_

#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "fcntl.h"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
class DebugTool {
public:
  DebugTool():fp(0) {}
  ~DebugTool() {
    close();
  }

  int open(const char* fn) {
    if (fp != 0) {
      fclose(fp);
    }

    fp = fopen(fn, "wb");
    if (fp == 0) {
      std::cout << "Error open file " << fn << "\n";
      return -1;
    }
    return 0;
  }

  void close() {
    if (fp != 0) {
      fclose(fp);
      fp = 0;
    }
  }

  int write_blob(const char* name, Blob<Dtype>* blob, bool write_diff) {
    return write_blob(string(name), blob, write_diff);
  }

  int write_blob(const string& name, Blob<Dtype>* blob, bool write_diff) {
    if (fp == 0) {
      return -1;
    }

    const int NAME_LEN = 240;
    char name_str[NAME_LEN];
    int num, channels, height, width;

    memset(name_str, 0, NAME_LEN);
    if (write_diff) {
      strncpy(name_str, (name+":diff").c_str(), NAME_LEN - 1);
    } else {
      strncpy(name_str, name.c_str(), NAME_LEN - 1);
    }

    fwrite(name_str, NAME_LEN, 1, fp);
    num = blob->num();
    fwrite(&num, sizeof(int), 1, fp);
    channels = blob->channels();
    fwrite(&channels, sizeof(int), 1, fp);  
    height = blob->height();
    fwrite(&height, sizeof(int), 1, fp);  
    width = blob->width();
    fwrite(&width, sizeof(int), 1, fp);
    
    const Dtype* data;
    if (write_diff) {
      data = blob->cpu_diff();
    } else {
      data = blob->cpu_data();
    }
    std::cout << blob->count();
    fwrite(data, sizeof(Dtype), blob->count(), fp);

    return 0;
  }

  FILE* fp;

};



INSTANTIATE_CLASS(DebugTool);
} // namespace caffe

#endif