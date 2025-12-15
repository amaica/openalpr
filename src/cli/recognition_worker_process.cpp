#include "recognition_worker_process.h"

#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <fcntl.h>
#include <signal.h>

#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "alpr.h"
#include "config.h"

namespace {

// Write the full buffer, handling partial writes.
bool writeAll(int fd, const void* buf, size_t len)
{
  const char* cbuf = static_cast<const char*>(buf);
  size_t written = 0;
  while (written < len)
  {
    ssize_t w = ::write(fd, cbuf + written, len - written);
    if (w <= 0) return false;
    written += static_cast<size_t>(w);
  }
  return true;
}

// Reads a single line (delimited by '\n') from fd.
bool readLine(int fd, std::string& out)
{
  out.clear();
  char ch;
  while (true)
  {
    ssize_t r = ::read(fd, &ch, 1);
    if (r == 0) return false;      // EOF
    if (r < 0) return false;       // error
    if (ch == '\n') break;
    out.push_back(ch);
  }
  return true;
}

} // namespace

RecognitionWorkerProcess::RecognitionWorkerProcess(const Params& params)
  : params_(params), childPid_(0), writeFd_(-1), readFd_(-1)
{
}

RecognitionWorkerProcess::~RecognitionWorkerProcess()
{
  stop();
}

bool RecognitionWorkerProcess::start()
{
  int toChild[2];
  int fromChild[2];
  if (pipe(toChild) != 0) return false;
  if (pipe(fromChild) != 0)
  {
    ::close(toChild[0]); ::close(toChild[1]);
    return false;
  }

  childPid_ = fork();
  if (childPid_ < 0)
  {
    ::close(toChild[0]); ::close(toChild[1]);
    ::close(fromChild[0]); ::close(fromChild[1]);
    return false;
  }

  if (childPid_ == 0)
  {
    // Child
    ::close(toChild[1]);
    ::close(fromChild[0]);

    int readFd = toChild[0];
    int writeFd = fromChild[1];

    alpr::Alpr alpr(params_.country, params_.configFile);
    alpr.setTopN(params_.topn);
    if (params_.detectRegion) alpr.setDetectRegion(true);
    if (!params_.templatePattern.empty()) alpr.setDefaultRegion(params_.templatePattern);
    if (params_.debug) alpr.getConfig()->setDebug(true);

    while (true)
    {
      std::string path;
      if (!readLine(readFd, path))
      {
        break;
      }
      if (path == "__quit")
      {
        break;
      }

      cv::Mat frame = cv::imread(path);
      if (frame.empty())
      {
        std::string msg = path + "\t{}\n";
        writeAll(writeFd, msg.data(), msg.size());
        continue;
      }

      std::vector<alpr::AlprRegionOfInterest> rois;
      rois.push_back(alpr::AlprRegionOfInterest(0, 0, frame.cols, frame.rows));

      alpr::AlprResults results = alpr.recognize(frame.data, frame.elemSize(), frame.cols, frame.rows, rois);
      std::string json = alpr.toJson(results);

      if (params_.measureProcessingTime && results.total_processing_time_ms > 0)
      {
        // Attach processing time as part of the JSON pipeline without altering schema.
      }

      std::string line = path + "\t" + json + "\n";
      writeAll(writeFd, line.data(), line.size());
    }

    ::close(readFd);
    ::close(writeFd);
    _exit(0);
  }

  // Parent
  ::close(toChild[0]);
  ::close(fromChild[1]);
  writeFd_ = toChild[1];
  readFd_ = fromChild[0];
  return true;
}

bool RecognitionWorkerProcess::sendJob(const std::string& imagePath)
{
  if (writeFd_ < 0) return false;
  std::string line = imagePath + "\n";
  return writeAll(writeFd_, line.data(), line.size());
}

bool RecognitionWorkerProcess::readResult(std::string& imagePath, std::string& jsonResult)
{
  if (readFd_ < 0) return false;
  std::string line;
  if (!readLine(readFd_, line)) return false;
  size_t tab = line.find('\t');
  if (tab == std::string::npos)
  {
    imagePath.clear();
    jsonResult = "{}";
    return true;
  }
  imagePath = line.substr(0, tab);
  jsonResult = line.substr(tab + 1);
  return true;
}

void RecognitionWorkerProcess::stop()
{
  if (childPid_ <= 0) return;

  if (writeFd_ >= 0)
  {
    std::string quit = "__quit\n";
    writeAll(writeFd_, quit.data(), quit.size());
    ::close(writeFd_);
    writeFd_ = -1;
  }
  if (readFd_ >= 0)
  {
    ::close(readFd_);
    readFd_ = -1;
  }

  int status = 0;
  waitpid(childPid_, &status, 0);
  childPid_ = 0;
}

