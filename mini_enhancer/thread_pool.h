#ifndef MINI_ENHANCER_THREAD_POOL_H
#define MINI_ENHANCER_THREAD_POOL_H

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// Minimal fixed-size thread pool (1–2 workers). No oversubscription.
class ThreadPool {
 public:
  explicit ThreadPool(std::size_t num_threads);
  ~ThreadPool();

  void enqueue(std::function<void()> task);
  void wait_all();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

 private:
  void worker_loop();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mtx_;
  std::condition_variable cv_task_;
  std::condition_variable cv_done_;
  bool stop_{false};
  std::size_t active_{0};
  std::size_t pending_{0};
};

inline ThreadPool::ThreadPool(std::size_t num_threads) {
  if (num_threads < 1) num_threads = 1;
  if (num_threads > 2) num_threads = 2;
  workers_.reserve(num_threads);
  for (std::size_t i = 0; i < num_threads; ++i)
    workers_.emplace_back(&ThreadPool::worker_loop, this);
}

inline ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    stop_ = true;
  }
  cv_task_.notify_all();
  for (auto& t : workers_)
    if (t.joinable()) t.join();
}

inline void ThreadPool::enqueue(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    tasks_.push(std::move(task));
    ++pending_;
  }
  cv_task_.notify_one();
}

inline void ThreadPool::wait_all() {
  std::unique_lock<std::mutex> lock(mtx_);
  cv_done_.wait(lock, [this] { return pending_ == 0 && active_ == 0; });
}

inline void ThreadPool::worker_loop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_task_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) return;
      if (tasks_.empty()) continue;
      task = std::move(tasks_.front());
      tasks_.pop();
      ++active_;
      --pending_;
    }
    if (task) task();
    {
      std::lock_guard<std::mutex> lock(mtx_);
      --active_;
    }
    cv_done_.notify_all();
  }
}

#endif
