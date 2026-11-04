#pragma once
// Bounded blocking FIFO used by the offline sequence runner (spec Section 15).
// push() blocks while full (backpressure; nothing is dropped). close() wakes
// all waiters; after close, push() returns false and pop() drains remaining
// items then returns nullopt. Capacity is fixed at construction.

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace framepipe::app {

template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) throw std::invalid_argument("BoundedQueue capacity must be positive");
    }

    bool push(T item) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return closed_ || q_.size() < capacity_; });
        if (closed_) return false;
        q_.push_back(std::move(item));
        if (q_.size() > high_water_) high_water_ = q_.size();
        not_empty_.notify_one();
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return std::nullopt;
        T v = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return v;
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        not_full_.notify_all();
        not_empty_.notify_all();
    }

    // Discards queued items (used on fail-fast cancellation).
    void clear() {
        std::lock_guard<std::mutex> lk(m_);
        q_.clear();
        not_full_.notify_all();
    }

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t high_water_mark() const {
        std::lock_guard<std::mutex> lk(m_);
        return high_water_;
    }

private:
    const std::size_t capacity_;
    mutable std::mutex m_;
    std::condition_variable not_full_, not_empty_;
    std::deque<T> q_;
    std::size_t high_water_ = 0;
    bool closed_ = false;
};

}  // namespace framepipe::app
