#include "daemon/process_worker_pool.h"

#include <unistd.h>
#include <sys/wait.h>
#include <poll.h>
#include <vector>

#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "openalpr/alpr.h"
#include "openalpr/config.h"

ProcessWorkerPool::ProcessWorkerPool(const ProcessWorkerParams& params, int workerCount)
  : params_(params), workerCount_(workerCount)
{
}

ProcessWorkerPool::~ProcessWorkerPool()
{
  stop();
}

bool ProcessWorkerPool::writeAll(int fd, const void* buf, size_t len)
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

bool ProcessWorkerPool::readAll(int fd, void* buf, size_t len)
{
  char* cbuf = static_cast<char*>(buf);
  size_t readBytes = 0;
  while (readBytes < len)
  {
    ssize_t r = ::read(fd, cbuf + readBytes, len - readBytes);
    if (r <= 0) return false;
    readBytes += static_cast<size_t>(r);
  }
  return true;
}

bool ProcessWorkerPool::start()
{
  workers_.resize(workerCount_);
  for (int i = 0; i < workerCount_; i++)
  {
    int toChild[2];
    int fromChild[2];
    if (pipe(toChild) != 0) return false;
    if (pipe(fromChild) != 0)
    {
      ::close(toChild[0]); ::close(toChild[1]);
      return false;
    }

    pid_t pid = fork();
    if (pid < 0)
    {
      ::close(toChild[0]); ::close(toChild[1]);
      ::close(fromChild[0]); ::close(fromChild[1]);
      return false;
    }

    if (pid == 0)
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
        uint32_t len = 0;
        if (!readAll(readFd, &len, sizeof(len)))
          break;
        if (len == 0)
          break; // shutdown

        std::vector<uchar> buffer(len);
        if (!readAll(readFd, buffer.data(), len))
          break;

        cv::Mat frame = cv::imdecode(buffer, 1);
        if (frame.empty())
        {
          uint32_t zero = 0;
          writeAll(writeFd, &zero, sizeof(zero));
          continue;
        }

        std::vector<alpr::AlprRegionOfInterest> rois;
        rois.push_back(alpr::AlprRegionOfInterest(0,0,frame.cols, frame.rows));
        alpr::AlprResults results = alpr.recognize(frame.data, frame.elemSize(), frame.cols, frame.rows, rois);
        std::string json = alpr.toJson(results);
        uint32_t outlen = static_cast<uint32_t>(json.size());
        writeAll(writeFd, &outlen, sizeof(outlen));
        writeAll(writeFd, json.data(), outlen);
      }

      ::close(readFd);
      ::close(writeFd);
      _exit(0);
    }

    // Parent
    ::close(toChild[0]);
    ::close(fromChild[1]);

    workers_[i].pid = pid;
    workers_[i].writeFd = toChild[1];
    workers_[i].readFd = fromChild[0];
    workers_[i].busy = false;
  }
  return true;
}

bool ProcessWorkerPool::dispatch(const cv::Mat& frame, const std::string& jobId)
{
  for (int i = 0; i < workerCount_; i++)
  {
    if (workers_[i].busy)
      continue;

    std::vector<uchar> buffer;
    if (!cv::imencode(".jpg", frame, buffer))
      return false;

    uint32_t len = static_cast<uint32_t>(buffer.size());
    if (!writeAll(workers_[i].writeFd, &len, sizeof(len)))
      return false;
    if (!writeAll(workers_[i].writeFd, buffer.data(), buffer.size()))
      return false;

    workers_[i].busy = true;
    workers_[i].jobId = jobId;
    workers_[i].frame = frame.clone();
    return true;
  }
  return false;
}

std::vector<ProcessWorkerPool::CompletedJob> ProcessWorkerPool::poll(int timeoutMs)
{
  std::vector<CompletedJob> completed;
  std::vector<pollfd> fds;
  std::vector<int> idxmap;

  for (int i = 0; i < workerCount_; i++)
  {
    if (!workers_[i].busy) continue;
    pollfd pfd;
    pfd.fd = workers_[i].readFd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    fds.push_back(pfd);
    idxmap.push_back(i);
  }

  if (fds.empty())
    return completed;

  int ret = ::poll(fds.data(), fds.size(), timeoutMs);
  if (ret <= 0)
    return completed;

  for (size_t i = 0; i < fds.size(); i++)
  {
    if (!(fds[i].revents & POLLIN))
      continue;
    int widx = idxmap[i];
    uint32_t len = 0;
    if (!readAll(workers_[widx].readFd, &len, sizeof(len)))
    {
      workers_[widx].busy = false;
      continue;
    }
    std::string json;
    if (len > 0)
    {
      json.resize(len);
      if (!readAll(workers_[widx].readFd, &json[0], len))
      {
        workers_[widx].busy = false;
        continue;
      }
    }

    CompletedJob job;
    job.jobId = workers_[widx].jobId;
    job.json = json;
    job.frame = workers_[widx].frame;
    workers_[widx].busy = false;
    workers_[widx].jobId.clear();
    workers_[widx].frame.release();
    completed.push_back(job);
  }

  return completed;
}

void ProcessWorkerPool::stop()
{
  for (int i = 0; i < workerCount_; i++)
  {
    if (workers_[i].writeFd >= 0)
    {
      uint32_t zero = 0;
      writeAll(workers_[i].writeFd, &zero, sizeof(zero));
      ::close(workers_[i].writeFd);
      workers_[i].writeFd = -1;
    }
    if (workers_[i].readFd >= 0)
    {
      ::close(workers_[i].readFd);
      workers_[i].readFd = -1;
    }
    if (workers_[i].pid > 0)
    {
      int status = 0;
      waitpid(workers_[i].pid, &status, 0);
      workers_[i].pid = 0;
    }
  }
}

