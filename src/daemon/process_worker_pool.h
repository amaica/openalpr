/*
 * Optional process-based worker pool for alprd.
 * Each worker owns its own Alpr instance; communication is via pipes.
 */

#pragma once

#include <string>
#include <vector>
#include <sys/types.h>
#include <opencv2/core/core.hpp>

namespace alpr
{
  class AlprResults;
}

struct ProcessWorkerParams
{
  std::string country;
  std::string configFile;
  std::string templatePattern;
  int topn = 10;
  bool detectRegion = false;
  bool debug = false;
};

class ProcessWorkerPool
{
public:
  explicit ProcessWorkerPool(const ProcessWorkerParams& params, int workerCount);
  ~ProcessWorkerPool();

  bool start();

  // Returns false if no worker is free.
  bool dispatch(const cv::Mat& frame, const std::string& jobId);

  struct CompletedJob
  {
    std::string jobId;
    std::string json;
    cv::Mat frame;
  };

  // Poll for completed jobs. timeoutMs can be zero.
  std::vector<CompletedJob> poll(int timeoutMs);

  void stop();

private:
  struct Worker
  {
    pid_t pid = 0;
    int writeFd = -1;
    int readFd = -1;
    bool busy = false;
    std::string jobId;
    cv::Mat frame;
  };

  ProcessWorkerParams params_;
  int workerCount_;
  std::vector<Worker> workers_;

  bool writeAll(int fd, const void* buf, size_t len);
  bool readAll(int fd, void* buf, size_t len);
};

