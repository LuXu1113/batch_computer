#pragma once

#include <pthread.h>

#include <string>
#include <vector>
#include <unordered_map>

#include "wolong/eigen/Dense"
#include "wolong/model_server/model/dnn_component.h"

namespace wolong {
namespace model_server {

class BatchComputerSummary {
  public:
    BatchComputerSummary();
    ~BatchComputerSummary();

    inline void record(const std::string &func, const double cost) {
      if ("request" == func) {
        total_request_time_ns_ += cost;
        total_request_counts_  += 1.0;
      } else if ("batch" == func) {
        total_batch_time_ns_ += cost;
        total_batch_counts_  += 1.0;
      }
    }

    inline double avg_time_cost(const std::string &func,
                                const std::string &unit) const {
      double ret = 0.0;
      if ("request" == func) {
        ret = total_request_time_ns_ / total_request_counts_;
      } else if ("batch" == func) {
        ret = total_batch_time_ns_ / total_batch_counts_;
      }

      if ("us" == unit) {
        ret /= 1e3;
      } else if ("ms" == unit) {
        ret /= 1e6;
      } else if ("sec" == unit) {
        ret /= 1e9;
      }
      return ret;
    }

    inline double avg_batch_size() const {
      return total_request_counts_ / total_batch_counts_;
    }

    inline void reset () {
      total_request_counts_  = 0.0;
      total_request_time_ns_ = 0.0;
      total_batch_counts_    = 0.0;
      total_batch_time_ns_   = 0.0;
    }

  private:
    double total_request_counts_;  // number of calls of predict()
    double total_request_time_ns_; // time cost in predict()
    double total_batch_counts_;    // number of calls of dnn_forward()
    double total_batch_time_ns_;   // time cost in dnn_forward()
};

class BatchComputer {
  public:
    BatchComputer();
    ~BatchComputer();

    int set_model(const std::vector<NetworkLayer> &layers);
    int predict(const Eigen::VectorXf &instance, double* dnn_q);
    void reset_summary();
    void print_summary() const;

  private:
    int add_task(const Eigen::VectorXf &instance, uint64 *runid);
    int run_task(const uint64 runid, double *dnn_q);
    int gen_runid(uint64 *runid);
    int get_result_by_runid(const uint64 runid, double *dnn_q);
    int batch_forward();
    void activation(Eigen::MatrixXf *neurons, const std::string &act_type);

    // dnn model
    std::vector<NetworkLayer> layers_;

    // intermediate variables,
    // avoid having to re-apply / release of memory for each call
    std::vector<Eigen::MatrixXf> activations_;

    // double bufferring
    Eigen::MatrixXf buffer_[2];
    std::vector<uint64> runid_[2];
    std::vector<const Eigen::VectorXf *>ins_ptr_[2];
    int running_;
    int waiting_;
    pthread_mutex_t running_mutex_;
    pthread_mutex_t waiting_mutex_;

    // results-Q,
    // key: runid; value: dnn output
    std::unordered_map<uint64, double> results_;

    // trace performace
    BatchComputerSummary summary_;
};

}  // end of namespace model_server
}  // end of namespace wolong


