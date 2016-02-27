#ifndef COMMON_H
#define COMMON_H

#include <memory>
#include <condition_variable>
#include <mutex>
#include <queue>

/* A simple threadsafe queue using a mutex and a condition variable. */
template <typename T>
class Queue
{
public:
    Queue() = default;

    Queue(Queue&& other)
    {
        std::unique_lock<std::mutex> lock(other.mutex_);
        queue_ = std::move(other.queue_);
    }

    void Push(T value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        lock.unlock();
        cond_.notify_one();
    }

    T Pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{return !queue_.empty();});
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    size_t Size()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_;
};

/* A pool of available contexts is simply implemented as a queue for our example. */
template <typename Context>
using ContextPool = Queue<std::unique_ptr<Context>>;

/* A RAII class for acquiring an execution context from a context pool. */
template <typename Context>
class ScopedContext
{
public:
    explicit ScopedContext(ContextPool<Context>& pool)
        : pool_(pool), context_(pool_.Pop())
    {
        context_->Activate();
    }

    ~ScopedContext()
    {
        context_->Deactivate();
        pool_.Push(std::move(context_));
    }

    Context* operator->() const
    {
        return context_.get();
    }

private:
    ContextPool<Context>& pool_;
    std::unique_ptr<Context> context_;
};

#endif // COMMON_H
