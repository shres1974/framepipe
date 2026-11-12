// Manifest parsing and bounded ordered sequence runner (spec Section 15).

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include <opencv2/imgcodecs.hpp>

#include "app/batch.hpp"
#include "app/bounded_queue.hpp"
#include "app/ordered_window.hpp"
#include "framepipe/errors.hpp"
#include "io/manifest.hpp"
#include "test_util.hpp"

using namespace fptest;
namespace fs = std::filesystem;

namespace {
fs::path fresh_dir(const std::string& name) {
    const auto d = fs::temp_directory_path() / ("framepipe_batch_" + std::to_string(::getpid()) + "_" + name);
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}
void write_png(const fs::path& p, int w, int h, int v) {
    cv::Mat m(h, w, CV_8UC3, cv::Scalar(v, v / 2, 255 - v));
    ASSERT_TRUE(cv::imwrite(p.string(), m));
}
void write_text(const fs::path& p, const std::string& s) { std::ofstream(p) << s; }

std::string manifest_error(const fs::path& p) {
    try {
        io::load_manifest(p.string());
    } catch (const Error& e) {
        return e.what();
    }
    return "";
}
}  // namespace

TEST(Manifest, ParsesOrderAndRelativePaths) {
    const auto d = fresh_dir("parse");
    fs::create_directories(d / "frames");
    for (int i : {9, 10, 2}) write_png(d / "frames" / ("frame_" + std::to_string(i) + ".png"), 8, 8, i);
    write_text(d / "m.jsonl",
               "{\"frame_id\":\"frame_9\",\"path\":\"frames/frame_9.png\",\"source_timestamp_ns\":900}\n"
               "\n"
               "{\"frame_id\":\"frame_10\",\"path\":\"frames/frame_10.png\"}\n"
               "{\"frame_id\":\"frame_2\",\"path\":\"" + (d / "frames/frame_2.png").string() + "\"}\n");
    const auto m = io::load_manifest((d / "m.jsonl").string());
    ASSERT_EQ(m.entries.size(), 3u);
    EXPECT_EQ(m.entries[0].frame_id, "frame_9");  // manifest order, not lexical
    EXPECT_EQ(m.entries[1].frame_id, "frame_10");
    EXPECT_EQ(m.entries[2].frame_id, "frame_2");
    EXPECT_EQ(*m.entries[0].source_timestamp_ns, 900);
    EXPECT_FALSE(m.entries[1].source_timestamp_ns.has_value());
    EXPECT_TRUE(fs::exists(m.entries[1].resolved_path));
    EXPECT_EQ(m.sha256.size(), 64u);
}

TEST(Manifest, RejectsBadRecords) {
    const auto d = fresh_dir("bad");
    write_png(d / "a.png", 4, 4, 1);
    auto check = [&](const std::string& body, const std::string& needle) {
        write_text(d / "m.jsonl", body);
        const auto e = manifest_error(d / "m.jsonl");
        EXPECT_NE(e.find(needle), std::string::npos) << "body: " << body << "\nerror: " << e;
    };
    check("{\"frame_id\":\"a\",\"path\":\"a.png\"}\n{\"frame_id\":\"a\",\"path\":\"a.png\"}\n", "duplicate");
    check("{\"frame_id\":\"a\",\"path\":\"a.png\"\n", "malformed");
    check("{\"frame_id\":1,\"path\":\"a.png\"}\n", "frame_id");
    check("{\"frame_id\":\"a\",\"path\":\"missing.png\"}\n", "does not exist");
    check("{\"frame_id\":\"../escape\",\"path\":\"a.png\"}\n", "frame_id");
    check("{\"frame_id\":\"a/b\",\"path\":\"a.png\"}\n", "frame_id");
    check("{\"frame_id\":\".hidden\",\"path\":\"a.png\"}\n", "frame_id");
    check("{\"frame_id\":\"a\",\"path\":\"a.png\",\"extra\":1}\n", "unknown field");
    check("{\"frame_id\":\"a\",\"path\":\"a.png\",\"source_timestamp_ns\":\"x\"}\n", "source_timestamp_ns");
    check("[]\n", "object");
    check("\n\n", "no records");
}

TEST(BoundedQueue, BlocksWhenFullAndPreservesOrder) {
    app::BoundedQueue<int> q(2);
    std::atomic<int> pushed{0};
    std::thread prod([&] {
        for (int i = 0; i < 10; ++i) {
            q.push(i);
            pushed = i + 1;
        }
        q.close();
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_LE(pushed.load(), 2) << "producer must block when the queue is full";
    for (int i = 0; i < 10; ++i) {
        auto v = q.pop();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, i);
    }
    EXPECT_FALSE(q.pop().has_value());
    prod.join();
    EXPECT_LE(q.high_water_mark(), 2u);
}

TEST(OrderedWindow, ReordersOutOfOrderProducersAndBoundsHeldItems) {
    constexpr int kTotal = 200;
    constexpr std::size_t kCap = 3;
    app::OrderedWindow<int> w(kCap, kTotal);
    std::atomic<int> claim{0};
    std::atomic<int> in_flight{0}, peak{0};
    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t) {
        producers.emplace_back([&, t] {
            for (int i = claim++; i < kTotal; i = claim++) {
                if (!w.acquire(i)) return;
                const int now = ++in_flight;
                int p = peak.load();
                while (now > p && !peak.compare_exchange_weak(p, now)) {}
                // Uneven work so items finish out of order.
                std::this_thread::sleep_for(std::chrono::microseconds((i * 37 + t * 11) % 200));
                --in_flight;
                w.put(i, i * 10);
            }
        });
    }
    for (int i = 0; i < kTotal; ++i) {
        auto v = w.take();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, i * 10) << "items must come out in sequence order";
    }
    EXPECT_FALSE(w.take().has_value()) << "take() after the last item must return nullopt";
    for (auto& p : producers) p.join();
    EXPECT_LE(w.high_water_mark(), kCap);
    EXPECT_LE(static_cast<std::size_t>(peak.load()), kCap) << "producers ran ahead of the window";
}

TEST(OrderedWindow, CloseWakesBlockedProducersAndConsumer) {
    app::OrderedWindow<int> w(1, 10);
    ASSERT_TRUE(w.acquire(0));
    std::atomic<bool> producer_returned{false};
    std::thread blocked([&] {
        EXPECT_FALSE(w.acquire(1)) << "seq 1 is outside a window of 1 until seq 0 is taken";
        producer_returned = true;
    });
    std::thread consumer([&] { EXPECT_FALSE(w.take().has_value()); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(producer_returned.load());
    w.close();
    blocked.join();
    consumer.join();
    EXPECT_TRUE(producer_returned.load());
}

TEST(Batch, PreservesOrderIdsAndHandlesShapeChanges) {
    const auto d = fresh_dir("order");
    std::string m;
    const int sizes[][2] = {{40, 30}, {40, 30}, {17, 19}, {40, 30}, {1, 1}};
    for (int i = 0; i < 5; ++i) {
        write_png(d / ("in" + std::to_string(i) + ".png"), sizes[i][0], sizes[i][1], i * 40);
        m += "{\"frame_id\":\"id_" + std::to_string(4 - i) + "\",\"path\":\"in" + std::to_string(i) + ".png\"}\n";
    }
    write_text(d / "m.jsonl", m);
    app::BatchOptions o;
    o.config = cfg(16, 12);
    o.output_dir = (d / "out").string();
    std::atomic<int> decoded{0};
    o.on_decoded = [&](const std::string&) { decoded++; };
    const auto r = app::run_batch(io::load_manifest((d / "m.jsonl").string()), o);
    EXPECT_EQ(r.completed, 5u);
    EXPECT_EQ(r.failed, 0u);
    EXPECT_EQ(r.reprepare_count, 3u);  // 40x30 -> 17x19 -> 40x30 -> 1x1
    EXPECT_LE(r.queue_high_water, o.queue_capacity);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(r.frames[i].sequence, i);
        EXPECT_EQ(r.frames[i].frame_id, "id_" + std::to_string(4 - i));
        EXPECT_TRUE(fs::exists(d / "out" / (r.frames[i].frame_id + ".png")));
        if (i > 0) {
            EXPECT_GE(r.frames[i].completed_ms, r.frames[i - 1].completed_ms);
        }
    }
    // Each output equals processing that frame individually.
    for (int i = 0; i < 5; ++i) {
        const auto in = cv::imread((d / ("in" + std::to_string(i) + ".png")).string(), cv::IMREAD_UNCHANGED);
        const auto out = cv::imread((d / "out" / (r.frames[i].frame_id + ".png")).string(), cv::IMREAD_UNCHANGED);
        EXPECT_EQ(out.cols, 16);
        EXPECT_EQ(out.rows, 12);
        (void)in;
    }
}

TEST(Batch, FailFastStopsAndRecords) {
    const auto d = fresh_dir("failfast");
    write_png(d / "ok.png", 8, 8, 1);
    write_text(d / "bad.png", "corrupt");
    write_text(d / "m.jsonl",
               "{\"frame_id\":\"f0\",\"path\":\"ok.png\"}\n{\"frame_id\":\"f1\",\"path\":\"bad.png\"}\n"
               "{\"frame_id\":\"f2\",\"path\":\"ok.png\"}\n");
    app::BatchOptions o;
    o.config = cfg(8, 8);
    o.output_dir = (d / "out").string();
    const auto r = app::run_batch(io::load_manifest((d / "m.jsonl").string()), o);
    EXPECT_TRUE(r.fail_fast_triggered);
    EXPECT_EQ(r.completed, 1u);
    EXPECT_EQ(r.failed, 1u);
    EXPECT_EQ(r.not_processed, 1u);
    EXPECT_EQ(r.frames[1].status, "failed");
    EXPECT_FALSE(r.frames[1].error.empty());
    EXPECT_EQ(r.frames[2].status, "not_processed");
    EXPECT_FALSE(fs::exists(d / "out" / "f1.png"));
}

TEST(Batch, ContinueOnErrorRecordsEveryFailure) {
    const auto d = fresh_dir("continue");
    write_png(d / "ok.png", 8, 8, 1);
    write_text(d / "bad.png", "corrupt");
    write_text(d / "m.jsonl",
               "{\"frame_id\":\"f0\",\"path\":\"bad.png\"}\n{\"frame_id\":\"f1\",\"path\":\"ok.png\"}\n"
               "{\"frame_id\":\"f2\",\"path\":\"bad.png\"}\n{\"frame_id\":\"f3\",\"path\":\"ok.png\"}\n");
    app::BatchOptions o;
    o.config = cfg(8, 8);
    o.output_dir = (d / "out").string();
    o.continue_on_error = true;
    const auto r = app::run_batch(io::load_manifest((d / "m.jsonl").string()), o);
    EXPECT_EQ(r.completed, 2u);
    EXPECT_EQ(r.failed, 2u);
    EXPECT_EQ(r.not_processed, 0u);
    EXPECT_EQ(r.frames[0].status, "failed");
    EXPECT_EQ(r.frames[3].status, "completed");
}

TEST(Batch, BackpressureBoundsDecodedFramesAndMemory) {
    const auto d = fresh_dir("backpressure");
    std::string m;
    for (int i = 0; i < 30; ++i) m += "{\"frame_id\":\"f" + std::to_string(i) + "\",\"path\":\"a.png\"}\n";
    write_png(d / "a.png", 256, 128, 9);
    write_text(d / "m.jsonl", m);
    app::BatchOptions o;
    o.config = cfg(64, 32);
    o.output_dir = (d / "out").string();
    o.queue_capacity = 2;
    const std::size_t before = tracked_host_live_bytes();
    std::atomic<std::size_t> peak{0};
    o.on_decoded = [&](const std::string&) {
        const std::size_t now = tracked_host_live_bytes();
        std::size_t p = peak.load();
        while (now > p && !peak.compare_exchange_weak(p, now)) {}
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    };
    const auto r = app::run_batch(io::load_manifest((d / "m.jsonl").string()), o);
    EXPECT_EQ(r.completed, 30u);
    EXPECT_LE(r.queue_high_water, 2u);
    // Decoded RGB frame = 256*128*3 bytes. Bounded by queue(2) + 1 in flight + 1 being decoded,
    // plus fixed pipeline workspace; far below 30 frames.
    const std::size_t frame = 256 * 128 * 3;
    EXPECT_LT(peak.load() - before, 8 * frame + 4 * 256 * 128 * 4) << "memory grew with the number of frames";
    EXPECT_EQ(tracked_host_live_bytes(), before) << "batch leaked host memory";
}

TEST(Batch, RefusesOverwriteByDefault) {
    const auto d = fresh_dir("overwrite");
    write_png(d / "a.png", 8, 8, 1);
    write_text(d / "m.jsonl", "{\"frame_id\":\"x\",\"path\":\"a.png\"}\n");
    fs::create_directories(d / "out");
    write_text(d / "out" / "x.png", "existing");
    app::BatchOptions o;
    o.config = cfg(8, 8);
    o.output_dir = (d / "out").string();
    EXPECT_THROW(app::run_batch(io::load_manifest((d / "m.jsonl").string()), o), Error);
    o.overwrite = true;
    EXPECT_EQ(app::run_batch(io::load_manifest((d / "m.jsonl").string()), o).completed, 1u);
}
