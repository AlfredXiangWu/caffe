#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include <fstream>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#define DEBUG 1
#include "caffe/debugtool.hpp"
#include <iostream>

namespace caffe {

using std::ifstream;

template <typename Dtype>
void TripletDataLayer<Dtype>::load_index(string fn)
{
  LOG(INFO) << "opening triplets file: " << fn;
  ifstream ifs(fn.c_str());
  if (!ifs) {
    LOG(FATAL) << "Error opening triplets list file: " << fn;
    return;
  }

  int num_triplets = 0;
  int idx1 = 0;

  ifs >> num_triplets;
  CHECK(num_triplets > 0);
  index_.resize(num_triplets * 3);
  int image_paths_size = image_paths_.size();
  
  // to do here

  for (int i = 0; i < num_triplets * 3; ++i) {
    if (!(ifs >> idx1)) {
      LOG(FATAL) << "Error reading triplets file: " << fn;
    }
    CHECK(idx1 < image_paths_size);
    index_[i] = idx1;
  }

  LOG(INFO) << "loading triplets done. Total triplets: " << index_.size() / 3;

#if 0 //DEBUG_WUHAO
/*
  for (int j = 0; j < 10; ++j){
    std::cout << pairs_[2 * j] << " " << pairs_[2 * j + 1] << "\n";
  }

  int pairs_size = pairs_.size();
  for (int j = 0; j < 10; ++j){
    std::cout << pairs_[pairs_size - 20 + 2 * j] << " " << pairs_[pairs_size - 20 + 2 * j + 1] << "\n";
  }
*/
#endif



}

template <typename Dtype>
void TripletDataLayer<Dtype>::load_list(string fn)
{
  LOG(INFO) << "opening image list file: " << fn;
  ifstream ifs(fn.c_str());
  if (!ifs) {
    LOG(FATAL) << "Error opening image list file: " << fn;
    return;
  }  

  string filename;
  int label;
  while(ifs >> filename >> label) {
    image_paths_.push_back(filename);
  }

  LOG(INFO) << "loading image list done. Total images: " << image_paths_.size();

#if 0 //DEBUG_WUHAO
/*
  std::cout << "image_paths.size(): " << image_paths_.size() << "\n";
  for (int j = 0; j < 10; ++j) {
    std::cout << image_paths_[j] << "\n";
  }
*/
#endif


}


template <typename Dtype>
TripletDataLayer<Dtype>::~TripletDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.triplet_data_param().backend()) {
  case PairDataParameter_DB_LEVELDB:
    break;  // do nothing
  case PairDataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void TripletDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.triplet_data_param().backend()) {
  case PairDataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.triplet_data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.triplet_data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.triplet_data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case PairDataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.triplet_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.triplet_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.triplet_data_param().backend()) {
  case PairDataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    iter_.reset();
    break;
  case PairDataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  int batch_size = this->layer_param_.triplet_data_param().batch_size();
  int channels = datum.channels();
  int height, width;

  if (crop_size > 0) {
    height = crop_size;
    width = crop_size;
  } else {
    height = datum.height();
    width = datum.width();
  }

  (*top)[0]->Reshape(batch_size, channels, height, width);
  (*top)[1]->Reshape(batch_size, channels, height, width);
  (*top)[2]->Reshape(batch_size, channels, height, width);  
  this->prefetch_data_.Reshape(batch_size, channels, height, width);
  this->prefetch_data2_.Reshape(batch_size, channels, height, width);
  this->prefetch_data3_.Reshape(batch_size, channels, height, width);

  this->prefetch_label_.Reshape(1,1,1,1);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();

  load_list(this->layer_param_.triplet_data_param().source_list());
  load_index(this->layer_param_.triplet_data_param().triplet_list());
  cur_ = 0;


}

template <typename Dtype>
void TripletDataLayer<Dtype>::get_cur_key(int index_channel, string& keystr){
  int idx =  0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  
  CHECK((index_channel == 1) || (index_channel == 0) || (index_channel == 2));
  idx = index_[cur_ * 3 + index_channel];

#if 0
  std::cout << "idx: " << idx << "image_paths_[idx]: " << image_paths_[idx] << std::endl;
#endif

  CHECK(idx < image_paths_.size());
  snprintf(key_cstr, kMaxKeyLength, "%08d_%s", idx,
      image_paths_[idx].c_str());
  keystr.clear();
  keystr = key_cstr;

}


template <typename Dtype>
void TripletDataLayer<Dtype>::get_value(string& keystr, Datum& datum){
  string value;
  leveldb::Status st;

  switch (this->layer_param_.triplet_data_param().backend()) {
  case PairDataParameter_DB_LEVELDB:
    //CHECK(iter_);
    //CHECK(iter_->Valid());

    st = db_->Get(leveldb::ReadOptions(), keystr, &value);
    CHECK(st.ok());
    datum.ParseFromString(value);
    break;
  case PairDataParameter_DB_LMDB:
    mdb_key_.mv_size = keystr.size();
    mdb_key_.mv_data = reinterpret_cast<void*>(&keystr[0]); 

#if 0 // DEBUG_WUHAO
    std::cout << "keystr: " << keystr << "\n";
    std::cout << "keystr.size(): " << keystr.size() << std::endl;
    std::cout << "mdb_key_.size: " << mdb_key_.mv_size << " mdb_key_.mv_data: " << (char*)mdb_key_.mv_data << std::endl;
#endif

    CHECK_EQ(mdb_get(mdb_txn_, mdb_dbi_, &mdb_key_, &mdb_value_), MDB_SUCCESS);   
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);  



    break;
  /*
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
            &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
    datum.ParseFromArray(mdb_value_.mv_data,
        mdb_value_.mv_size);
    break;
  */
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void TripletDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_data2 = this->prefetch_data2_.mutable_cpu_data();  
  Dtype* top_data3 = this->prefetch_data3_.mutable_cpu_data();
  //Dtype* top_label2 = this->prefetch_label2_.mutable_cpu_data();

  const int batch_size = this->layer_param_.triplet_data_param().batch_size();
  string keystr, keystr2, keystr3;
  int label, label2, label3;

  for (int item_id = 0; item_id < batch_size; ++item_id) {

    get_cur_key(0, keystr);
    get_value(keystr, datum);
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);
    label = datum.label();
    
    get_cur_key(1, keystr2);
    get_value(keystr2, datum);
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data2);
    label2 = datum.label();

    get_cur_key(2, keystr3);
    get_value(keystr3, datum);
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data3);    
    label3 = datum.label();

#if 0
    if (Caffe::phase() == Caffe::TRAIN) {   
      LOG(INFO) << "Caffe phase: TRAIN";   
    } else {
      LOG(INFO) << "Caffe phase: TEST";
    }

    LOG(INFO) << " labels: " << label << " "<< label2  <<" " << label3;
#endif

    if ((label != label2) || (label == label3)) {
      LOG(INFO) << "anchor, pos, neg: label error";
      LOG(INFO) << "  keystr1: " << keystr << " label1: " << label;
      LOG(INFO) << "  keystr2: " << keystr2 << " label2: " << label2;     
      LOG(INFO) << "  keystr3: " << keystr3 << " label3: " << label3;

    }
    // go to the next iter

    cur_++;
    if (cur_ > index_.size() / 3 - 1) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cur_ = 0;
    }
  }


#if 0 // DEBUG_WUHAO
/*  
  DebugTool<Dtype> dbg;
  dbg.open("data.bin");
  dbg.write_blob("data", this->prefetch_data_, 0);
  dbg.write_blob("data2", this->prefetch_data2_, 0);
  dbg.write_blob("label", this->prefetch_label_, 0);
  dbg.write_blob("label2", this->prefetch_label2_, 0);  
  dbg.close();

  string str;
  std::cout << std::flush << "pause..." << std::flush;
  std::cin >> str;
*/  
#endif

}




template <typename Dtype>
void TripletDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(this->prefetch_data2_.count(), this->prefetch_data2_.cpu_data(),
             (*top)[1]->mutable_cpu_data());
  caffe_copy(this->prefetch_data3_.count(), this->prefetch_data3_.cpu_data(),
             (*top)[2]->mutable_cpu_data());
 

#if 0 // DEBUG_WUHAO
  
  LOG(INFO) << "in TripletDataLayer::Forward_cpu ";

  DebugTool<Dtype> dbg;
  dbg.open("data.bin");
  dbg.write_blob("data", *(*top)[0], 0);
  dbg.write_blob("data2", *(*top)[1], 0);
  dbg.write_blob("data3", *(*top)[2], 0);
  dbg.close();

  string str;
  std::cout << std::flush << "pause..." << std::flush;
  std::cin >> str;
  
#endif

  // Start a new prefetch thread
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
}



#ifdef CPU_ONLY
STUB_GPU_FORWARD(TripletDataLayer, Forward);
#endif

INSTANTIATE_CLASS(TripletDataLayer);

}  // namespace caffe
