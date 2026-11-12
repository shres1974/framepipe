#pragma once
// Bounded reorder window for the batch runner's parallel decode stage.
//
// Several producers decode frames out of order; one consumer takes them back
// strictly in sequence order. A producer must acquire() a sequence number
// before doing any work for it, and acquire() blocks while that number is
// `capacity` or more ahead of the consumer. So at most `capacity` items are
// held (decoding or decoded but not yet taken) no matter how many producers
// run. close() cancels: it wakes every waiter and discards held items.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace framepipe::app {

template <class T>
class OrderedWindow {
public:
    OrderedWindow(std::size_t capacity, std::int64_t total) : capacity_(capacity), total_(total) {
        if (capacity == 0) throw std::invalid_argument("OrderedWindow capacity must be positive");
    }

    // Blocks until `seq` fits in the window. Returns false once closed.
    bool acquire(std::int64_t seq) {
        std::unique_lock<std::mutex> lk(m_);
        slot_free_.wait(lk, [&] { return closed_ || seq < next_ + static_cast<std::int64_t>(capacity_); });
        if (closed_) return false;
        const auto held = static_cast<std::size_t>(seq - next_ + 1);
        if (held > high_water_) high_water_ = held;
        return true;
    }

    // Publishes the item for an acquired `seq`. Dropped if already closed.
    void put(std::int64_t seq, T item) {
        std::lock_guard<std::mutex> lk(m_);
        if (closed_) return;
        items_.emplace(seq, std::move(item));
        if (seq == next_) ready_.notify_one();
    }

    // Returns the next item in sequence order. nullopt after all `total`
    // items were taken, or once closed.
    std::optional<T> take() {
        std::unique_lock<std::mutex> lk(m_);
        ready_.wait(lk, [&] { return closed_ || next_ >= total_ || items_.count(next_) != 0; });
        if (closed_ || next_ >= total_) return std::nullopt;
        auto it = items_.find(next_);
        T v = std::move(it->second);
        items_.erase(it);
        ++next_;
        slot_free_.notify_all();
        return v;
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        items_.clear();
        slot_free_.notify_all();
        ready_.notify_all();
    }

    std::size_t capacity() const noexcept { return capacity_; }
    // Largest number of sequence numbers held at once (acquired, not yet taken).
    std::size_t high_water_mark() const {
        std::lock_guard<std::mutex> lk(m_);
        return high_water_;
    }

private:
    const std::size_t capacity_;
    const std::int64_t total_;
    mutable std::mutex m_;
    std::condition_variable slot_free_, ready_;
    std::map<std::int64_t, T> items_;
    std::int64_t next_ = 0;
    std::size_t high_water_ = 0;
    bool closed_ = false;
};

}  // namespace framepipe::app
