//
//  Queue.hpp
//  ParticleSphereCPU_threaded
//
//  Created by Shamyl Zakariya on 6/12/19.
//

#ifndef Queue_hpp
#define Queue_hpp

#include <condition_variable>
#include <mutex>
#include <queue>

// A threadsafe-queue adapted from https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
template <class T>
class SafeQueue {
public:
    SafeQueue(void)
        : _queue()
        , _mutex()
        , _condition()
        , _maxSize(0)
    {
    }

    SafeQueue(size_t maxSize)
        : _queue()
        , _mutex()
        , _condition()
        , _maxSize(maxSize)
    {
    }

    void setMaxSize(size_t maxSize)
    {
        _maxSize = maxSize;
    }

    size_t getMaxSize() const { return _maxSize; }

    // Add an element to the queue.
    void enqueue(T t)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push(t);
        if (_maxSize > 0) {
            while (_queue.size() > _maxSize) {
                _queue.pop();
            }
        }
        _condition.notify_one();
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.empty();
    }

    bool peek(T& t) const
    {
        std::unique_lock<std::mutex> lock(_mutex);
        if (!_queue.empty()) {
            t = _queue.back();
            return true;
        }
        return false;
    }

    // Get the "front"-element.
    // If the queue is empty, wait till a element is avaiable.
    T dequeue(void)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        while (_queue.empty()) {
            // release lock as long as the wait and reaquire it afterwards.
            _condition.wait(lock);
        }
        T val = _queue.front();
        _queue.pop();
        return val;
    }

    size_t size() const
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _queue.size();
    }

private:
    std::queue<T> _queue;
    mutable std::mutex _mutex;
    std::condition_variable _condition;
    size_t _maxSize;
};

#endif /* Queue_hpp */
