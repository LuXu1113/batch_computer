#include "wolong/model_server/model/batch_computer.h"

#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>

#include "base/common/base.h"
#include "base/common/logging.h"

using std::string;
using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::NoChange_t;

namespace wolong {
namespace model_server {

BatchComputerSummary::BatchComputerSummary()
  : total_request_counts_(0.0),
    total_request_time_ns_(0.0),
    total_batch_counts_(0.0),
    total_batch_time_ns_(0.0) {
}

BatchComputerSummary::~BatchComputerSummary() {
}

BatchComputer::BatchComputer()
  : running_(0),
    waiting_(1) {
  pthread_mutex_init(&running_mutex_, 0);
  pthread_mutex_init(&waiting_mutex_, 0);
}

BatchComputer::~BatchComputer() {
  pthread_mutex_destroy(&running_mutex_);
  pthread_mutex_destroy(&waiting_mutex_);
}

#define BUFFER_LEN (128)
int BatchComputer::set_model(const vector<NetworkLayer> &layers) {
  int rtv = 0;

  pthread_mutex_lock(&(running_mutex_));
  pthread_mutex_lock(&(waiting_mutex_));

  // copy layers to layers_,
  // transpose W, as it is faster using mkl.
  layers_.clear();
  int n_layers = layers.size();
  for (int i = 0; i < n_layers; ++i) {
    layers_.push_back(layers[i]);
    layers_[i].W = layers[i].W.transpose();
  }

  // initialize activations
  activations_.reserve(n_layers);
  for (int i = 0; i < n_layers; ++i) {
    activations_.push_back(MatrixXf(BUFFER_LEN, layers_[i].out_size));
  }

  // initialize double buffers
  buffer_[0].resize(BUFFER_LEN, layers_[0].in_size);
  buffer_[1].resize(BUFFER_LEN, layers_[0].in_size);
  runid_[0].clear();
  runid_[1].clear();
  ins_ptr_[0].clear();
  ins_ptr_[1].clear();
  running_ = 0;
  waiting_ = 1;

  // clear unuse results
  results_.clear();

  pthread_mutex_unlock(&(waiting_mutex_));
  pthread_mutex_unlock(&(running_mutex_));

  mkl_set_num_threads(6);

  return rtv;
}
#undef BUFFER_LEN

int BatchComputer::predict(const VectorXf &instance, double* dnn_q) {
  int rtv = 0;
  struct timespec ts1 = {0, 0};
  struct timespec ts2 = {0, 0};
  double exec_ns = 0.0;

  clock_gettime(CLOCK_MONOTONIC, &ts1);
  do {
    uint64 runid = 0;
    rtv = add_task(instance, &runid);
    if (0 != rtv) {
      LOG(WARNING) << "add_task() fail, rtv = " << rtv;
      break;
    }

    rtv = run_task(runid, dnn_q);
    if (0 != rtv) {
      LOG(WARNING) << "run_task() fail, "
                   << "runid = " << runid
                   << ", rtv = " << rtv;
      break;
    }
  } while (0);
  clock_gettime(CLOCK_MONOTONIC, &ts2);
  exec_ns = 1e9 * (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec);
  summary_.record(string("request"), exec_ns);

  return rtv;
}

int BatchComputer::add_task(const Eigen::VectorXf &instance, uint64 *runid) {
  int rtv = 0;
  int row_id = 0;
  
  pthread_mutex_lock(&(waiting_mutex_));
  do {
    // applay for a runid
    rtv = gen_runid(runid);
    if (0 != rtv) {
      LOG(WARNING) << "gen_runid() fail, rtv = " << rtv;
      break;
    }

    // put instance and its runid input waiting buffer
    runid_[waiting_].push_back(*runid);
    ins_ptr_[waiting_].push_back(&instance);
  } while (0);
  pthread_mutex_unlock(&(waiting_mutex_));

  return rtv;
}

#define EPSILON (1e-6)
#define INVALID (1e9)
int BatchComputer::run_task(const uint64 runid, double *dnn_q) {
  int rtv = 0;

  pthread_mutex_lock(&(running_mutex_));
  do {
    if (fabs(results_[runid] - INVALID) < EPSILON) {
      // switch running and waiting buffer
      pthread_mutex_lock(&(waiting_mutex_));
      running_ = 1 - running_;
      waiting_ = 1 - running_;
      runid_[waiting_].clear();
      ins_ptr_[waiting_].clear();
      pthread_mutex_unlock(&(waiting_mutex_));

      // expand buffer when it is full and a new task come
      int top = runid_[running_].size();
      if (top > buffer_[running_].rows()) {
        buffer_[running_].resize(top * 2, Eigen::NoChange);
      }

      // copy instances to batch matrix
      for (int i = 0; i < top; ++i) {
        buffer_[running_].row(i) = *(ins_ptr_[running_][i]);
      }

      // compute dnn
      rtv = batch_forward();
      if (0 != rtv) {
        LOG(WARNING) << "batch_forward() fail, rtv = " << rtv;
        break;
      }
    }
  } while (0);
  pthread_mutex_unlock(&(running_mutex_));

  if (0 == rtv) {
    // fetch result by runid
    rtv = get_result_by_runid(runid, dnn_q);
    if (0 != rtv) {
        LOG(WARNING) << "get_result_by_runid() fail, rtv = " << rtv;
    }
  }

  return rtv;
}

int BatchComputer::gen_runid(uint64 *runid) {
  int rtv = 0;

  uint64 new_id = syscall(SYS_gettid);
  if (results_.find(new_id) == results_.end()) {
    results_.insert(std::make_pair(new_id, INVALID));
  } else {
    results_[new_id] = INVALID;
  }
  (*runid) = new_id;

  return rtv;
}

int BatchComputer::get_result_by_runid(const uint64 runid, double *dnn_q) {
  int rtv = 0;

  auto res = results_.find(runid);
  if (results_.end() != res) {
    (*dnn_q) = res->second;
  } else {
    LOG(WARNING) << "runid losts, runid = " << runid << ".";
    rtv = -2;
  }

  return rtv;
}

#undef INVALID
#undef EPSILON

int BatchComputer::batch_forward() {
  int rtv = 0;
  struct timespec ts1 = {0, 0};
  struct timespec ts2 = {0, 0};
  double exec_ns = 0.0;

  clock_gettime(CLOCK_MONOTONIC, &ts1);
  do {
    int depth = layers_.size();
    if (depth <= 0) {
      LOG(WARNING) << "invalid dnn depth, depth = " << depth;
      rtv = -3;
      break;
    }

    int batch_size = runid_[running_].size();
    if (batch_size <= 0) {
      LOG(WARNING) << "invalid batch size, batch_size = " << batch_size;
      rtv = -4;
      break;
    }

    int compute_batch_size = (batch_size > 10 ? batch_size : 10);

    for (int i = 0; i < depth; ++i) {
      activations_[i].resize(compute_batch_size, Eigen::NoChange);
      activations_[i].setZero();
    }

    // first layer
    int n_dim = buffer_[running_].cols();
    if (layers_[0].use_bias) {
      activations_[0].noalias() = buffer_[running_].block(0, 0, compute_batch_size, n_dim) * layers_[0].W;
      activations_[0].rowwise() += layers_[0].b.transpose();
    } else {
      activations_[0].noalias() = buffer_[running_].block(0, 0, compute_batch_size, n_dim) * layers_[0].W;
    }
    activation(&(activations_[0]), layers_[0].act_type);

    // other layers
    for (int i = 1; i < depth; ++i) {
      if (layers_[i].use_bias) {
        activations_[i].noalias() = activations_[i - 1] * layers_[i].W;
        activations_[i].rowwise() += layers_[i].b.transpose();
      } else {
        activations_[i].noalias() = activations_[i - 1] * layers_[i].W;
      }
      activation(&(activations_[i]), layers_[i].act_type);
    }

    // put dnn-q into results_
    int last = depth - 1;
    for (int i = 0; i < batch_size; ++i) {
      results_[runid_[running_][i]] = activations_[last](i, 0);
    }
  } while (0);
  clock_gettime(CLOCK_MONOTONIC, &ts2);
  exec_ns = 1e9 * (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec);
  summary_.record(string("batch"), exec_ns);

  return rtv;
}

void BatchComputer::activation(Eigen::MatrixXf *neurons, const string &act_type) {
  if (act_type == "relu") {
    (*neurons) = neurons->cwiseMax(0);
  } else if (act_type == "sigmoid") {
    (*neurons) = 1 / (1 + (-(*neurons)).array().exp());
  } else {
    LOG(WARNING) << "invalid activation type, act_type = " << act_type;
  }
}

void BatchComputer::reset_summary() {
  summary_.reset();
}

void BatchComputer::print_summary() const {
  LOG(INFO) << "Avg-cost(request): " << summary_.avg_time_cost(string("request"), string("ms")) << " ms.";
  LOG(INFO) << "Avg-cost(batch): " << summary_.avg_time_cost(string("batch"), string("ms")) << " ms.";
  LOG(INFO) << "Avg-batchsize: " << summary_.avg_batch_size();
}

}  // end of namespace model_server
}  // end of namespace wolong

