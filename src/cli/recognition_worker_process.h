/*
 * Lightweight process-based worker used by the CLI to parallelize
 * image recognition without sharing Alpr instances across threads.
 */

#pragma once

#include <string>
#include <vector>
#include <sys/types.h>

namespace alpr
{
  class Alpr;
}

class RecognitionWorkerProcess
{
public:
  struct Params
  {
    std::string country;
    std::string configFile;
    std::string templatePattern;
    int topn = 10;
    bool detectRegion = false;
    bool debug = false;
    bool measureProcessingTime = false;
  };

  explicit RecognitionWorkerProcess(const Params& params);
  ~RecognitionWorkerProcess();

  // Forks the worker process and initializes IPC pipes.
  bool start();

  // Sends a single image path to the worker. Returns false on IPC error.
  bool sendJob(const std::string& imagePath);

  // Reads one result from the worker. Returns false on EOF or error.
  bool readResult(std::string& imagePath, std::string& jsonResult);

  // Gracefully stops the worker (sends quit signal and waits).
  void stop();

  bool isRunning() const { return childPid_ > 0; }
  int readFd() const { return readFd_; }

private:
  Params params_;
  pid_t childPid_;
  int writeFd_; // parent -> child
  int readFd_;  // child -> parent
};

