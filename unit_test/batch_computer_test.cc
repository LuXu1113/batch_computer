#include <time.h>
#include <pthread.h>
#include <string>
#include <vector>
#include "base/testing/gtest.h"
#include "base/common/base.h"
#include "base/common/gflags.h"
#include "base/common/logging.h"
#include "wolong/eigen/Dense"

#define private public
#include "wolong/model_server/model/batch_computer.h"

using std::string;
using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using wolong::model_server::NetworkLayer;
using wolong::model_server::BatchComputer;

class ElsDnnTest : public testing::Test {
  protected:
    static void SetUpTestCase() {
      NetworkLayer layer_0 = {
        .in_size     = 1455,
        .out_size    = 511,
        .use_bias    = true,
        .layer_index = 0,
        .act_tyep    = "relu"
      };
      layer_0.W = MatrixXf::Random(layer_0.out_size, layer_0.in_size);
      layer_0.b = MatrixXf::Random(layer_0.out_size, 1);

      NetworkLayer layer_1 = {
        .in_size     = 511,
        .out_size    = 255,
        .use_bias    = true,
        .layer_index = 1,
        .act_tyep    = "relu"
      };
      layer_1.W = MatrixXf::Random(layer_1.out_size, layer_1.in_size);
      layer_1.b = MatrixXf::Random(layer_1.out_size, 1);

      NetworkLayer layer_2 = {
        .in_size     = 255,
        .out_size    = 127,
        .use_bias    = true,
        .layer_index = 2,
        .act_tyep    = "relu"
      };
      layer_2.W = MatrixXf::Random(layer_2.out_size, layer_2.in_size);
      layer_2.b = MatrixXf::Random(layer_2.out_size, 1);

      NetworkLayer layer_3 = {
        .in_size     = 127,
        .out_size    = 127,
        .use_bias    = true,
        .layer_index = 3,
        .act_tyep    = "relu"
      };
      layer_3.W = MatrixXf::Random(layer_3.out_size, layer_3.in_size);
      layer_3.b = MatrixXf::Random(layer_3.out_size, 1);

      NetworkLayer layer_4 = {
        .in_size     = 127,
        .out_size    = 127,
        .use_bias    = true,
        .layer_index = 4,
        .act_tyep    = "relu"
      };
      layer_4.W = MatrixXf::Random(layer_4.out_size, layer_4.in_size);
      layer_4.b = MatrixXf::Random(layer_4.out_size, 1);

      NetworkLayer layer_5 = {
        .in_size     = 127,
        .out_size    = 1,
        .use_bias    = true,
        .layer_index = 4,
        .act_type    = "sigmoid"
      };
      layer_5.W = MatrixXf::Random(layer_5.out_size, layer_5.in_size);
      layer_5.b = MatrixXf::Random(layer_5.out_size, 1);

      model_.clear();
      model_.push_back(layer_0);
      model_.push_back(layer_1);
      model_.push_back(layer_2);
      model_.push_back(layer_3);
      model_.push_back(layer_4);
      model_.push_back(layer_5);
    }

    static void TearDownTestCase() {
      std::vector<NetworkLayer>().swap(model_);
    }

    static vector<NetworkLayer> model_;
};
vector<NetworkLayer> ElsDnnTest::model_;

TEST_F(ElsDnnTest, set_model) {
  int rtv = 0;
  BatchComputer bc;

  rtv = bc.set_model(model_);
  EXPECT_EQ(0, rtv);

  // check layers
  EXPECT_EQ(model_.size(), bc.layers_.size());
  for (int i = 0; i < model_.size(); ++i) {
    EXPECT_EQ(model_[i].in_size, bc.layers_[i].in_size);
    EXPECT_EQ(model_[i].out_size, bc.layers_[i].out_size);
    EXPECT_EQ(model_[i].use_bias, bc.layers_[i].use_bias);
    EXPECT_EQ(model_[i].layer_index, bc.layers_[i].layer_index);
    EXPECT_STREQ(model_[i].act_type.c_str(), bc.layers_[i].act_type.c_str());

    for (int j = 0; j < model_[i].in_size; ++j) {
      for (int k = 0; k < model_[i].out_size; ++k) {
        EXPECT_FLOAT_EQ(model_[i].W(k, j), bc.layers_[i].W(j, k));
      }
    }

    for (int j = 0; j < model_[i].out_size; ++j) {
      EXPECT_FLOAT_EQ(model_[i].b(j), bc.layers_[i].b(j));
    }
  }

  EXPECT_EQ(0, bc.runid_[0].size());
  EXPECT_EQ(0, bc.runid_[1].size());
  EXPECT_EQ(0, bc.running_);
  EXPECT_EQ(1, bc.waiting_);
  EXPECT_EQ(0, bc.results_.size());

  EXPECT_DOUBLE_EQ(0, bc.summary_.total_request_counts_);
  EXPECT_DOUBLE_EQ(0, bc.summary_.total_request_time_ns_);
  EXPECT_DOUBLE_EQ(0, bc.summary_.total_batch_counts_);
  EXPECT_DOUBLE_EQ(0, bc.summary_.total_batch_time_ns_);
}

#define NUM_TEST_ROUNDS_PER_THREAD (256)

TEST_F(ElsDnnTest, single_thread_perf) {
  int rtv = 0;

  BatchComputer bc;
  rtv = bc.set_model(model_);
  EXPECT_EQ(0, rtv);

  struct timespec ts1 = {0, 0};
  struct timespec ts2 = {0, 0};
  double exec_ms = 0.0;

  VectorXf instance = MatrixXf::Random(model_[0].in_size, 1);
  double prediction;

  clock_gettime(CLOCK_MONOTONIC, &ts1);
  for (int i = 0; i < NUM_TEST_ROUNDS_PER_THREAD; ++i) {
    rtv = bc.predict(instance, &prediction);
    EXPECT_EQ(0, rtv);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts2);
  exec_ms = 1e3 * (ts2.tv_sec - ts1.tv_sec) + 1e-6 * (ts2.tv_nsec - ts1.tv_nsec);
  exec_ms /= NUM_TEST_ROUNDS_PER_THREAD;

  EXPECT_GT(2, exec_ms);

  LOG(INFO) << "average prediction cost(single-thread): " << exec_ms << " ms.";
}

void *predict(void *arg) {
  int rtv = 0;
  BatchComputer *bc = (BatchComputer *)arg;

  VectorXf instance = MatrixXf::Random(bc->layers_[0].in_size, 1);
  double prediction;

  for (int i = 0; i < NUM_TEST_ROUNDS_PER_THREAD; ++i) {
    rtv = bc->predict(instance, &prediction);
    EXPECT_EQ(0, rtv);
  }

  return NULL;
}

TEST_F(ElsDnnTest, multi_thread_perf) {
  int rtv = 0;

  BatchComputer bc;
  rtv = bc.set_model(model_);
  EXPECT_EQ(0, rtv);

  for (int n_thread = 8; n_thread <= 64; n_thread += 8) {
    struct timespec ts1 = {0, 0};
    struct timespec ts2 = {0, 0};
    double exec_ms = 0.0;

    static pthread_t tid[64];
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    for (int i = 0; i < n_thread; ++i) {
      rtv = pthread_create(&tid[i], NULL, predict, &bc);
      EXPECT_EQ(0, rtv);
    }
    for (int i = 0; i < n_thread; ++i) {
      pthread_join(tid[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    exec_ms = 1e3 * (ts2.tv_sec - ts1.tv_sec) + 1e-6 * (ts2.tv_nsec - ts1.tv_nsec);
    exec_ms /= NUM_TEST_ROUNDS_PER_THREAD * n_thread;

    EXPECT_GT(2, exec_ms);
    LOG(INFO) << "average prediction cost(" << n_thread << " thread): " << exec_ms << " ms.";
  }
}

TEST_F(ElsDnnTest, multi_thread_perf_contrast) {
  int rtv = 0;

  mkl_set_num_threads(1);
  BatchComputer bc[64];
  for (int i = 0; i < 64; ++i) {
    rtv = bc[i].set_model(model_);
    EXPECT_EQ(0, rtv);
  }

  for (int n_thread = 8; n_thread <= 64; n_thread += 8) {
    struct timespec ts1 = {0, 0};
    struct timespec ts2 = {0, 0};
    double exec_ms = 0.0;

    static pthread_t tid[64];
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    for (int i = 0; i < n_thread; ++i) {
      rtv = pthread_create(&tid[i], NULL, predict, &bc[i]);
      EXPECT_EQ(0, rtv);
    }
    for (int i = 0; i < n_thread; ++i) {
      pthread_join(tid[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    exec_ms = 1e3 * (ts2.tv_sec - ts1.tv_sec) + 1e-6 * (ts2.tv_nsec - ts1.tv_nsec);
    exec_ms /= NUM_TEST_ROUNDS_PER_THREAD * n_thread;

    EXPECT_GT(2, exec_ms);
    LOG(INFO) << "average prediction cost(" << n_thread << " thread contrast): " << exec_ms << " ms.";
  }
}

int main (int argc, char **argv) {
  base::InitApp(&argc, &argv, "");
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

