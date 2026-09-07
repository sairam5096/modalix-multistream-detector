// Copyright 2026 SiMa Technologies, Inc.
//
// Shared-model multi-stream object detector (C++). Several object-detection
// models are loaded ONCE as SHARED detector stages on the single accelerator;
// every camera fans into the model(s) it is routed to, using the same realtime
// fan-in the high-density example uses to reach 48 streams. Each camera carries
// a model SET: one = route, several = chain (detections merged, labelled by
// model). A small HTTP API (+ web UI) lets the model of any camera be switched
// live; switching rebuilds the single shared graph. A watchdog self-heals a
// stalled or collapsed pipeline.
//
// C++ (no GIL) so the pull loop keeps up with a full-rate accelerator.

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/object_detection/obj_detection_utils.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace neat = simaai::neat;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::atomic<int> g_stop{0};
void on_signal(int) { g_stop.store(1); }

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- config

struct ModelEntry {
  std::string name;
  std::string path;
  std::string decode_type = "yolo26";
  std::string labels_path;
  double min_score = 0.45;
  double nms_iou = 0.60;
  int max_detections = 50;
  std::string description;
  std::vector<std::string> class_filter;  // lower-cased; empty = all
};

struct StreamCfg {
  std::string url;
  std::vector<std::string> models;
};

struct AppConfig {
  std::vector<ModelEntry> models;
  std::vector<StreamCfg> streams;
  int input_width = 1280, input_height = 720, input_fps = 30, target_fps = 0;
  int latency_ms = 100, decoder_buffers = 4, decoder_input_buffers = 1;
  std::string decoder_tuning = "low-memory";
  int queue_depth = 4, internal_queue_depth = 1;
  int max_inflight_per_stream = 2, max_inflight_total = 8;
  double watchdog_stall_s = 20.0;
  int rebuild_retries = 5;
  double report_hz = 10.0;
  bool watchdog_enabled = true;
  // ---- crops (exact SOM-side cropping) ----
  bool crops_enabled = false;
  std::string crops_dir = "/media/nvme/multimodel/crops";
  double crop_min_conf = 0.45;
  double crop_dedup_iou = 0.5;
  double crop_dedup_window = 12.0;
  double crop_pad = 0.08;
  int crop_max_disk = 6000;
  // Downscale the crop frame-tee to save CMA at high stream counts (0 = full res).
  // The teed frame is resized to tee_w x tee_h; detection boxes are scaled to match.
  int crop_tee_w = 0, crop_tee_h = 0;
  int crop_workers = 3;  // threads doing NV12->BGR + JPEG encode, off the pump
  // Which streams get the crop frame-tee. Empty = all. At high stream counts the
  // board cannot egress every stream's frames AND detect on all of them, so crops
  // run on a subset while detection stays on all streams.
  std::vector<int> crop_streams;
  std::string control_host = "0.0.0.0";
  int control_port = 8600;
};

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

fs::path resolve(const fs::path& base, const std::string& v) {
  fs::path p(v);
  return p.is_absolute() ? p : base.parent_path() / p;
}

std::vector<std::string> as_str_list(const YAML::Node& n) {
  std::vector<std::string> out;
  if (!n) return out;
  if (n.IsScalar()) { out.push_back(n.as<std::string>()); return out; }
  for (const auto& e : n) out.push_back(e.as<std::string>());
  return out;
}

AppConfig load_config(const fs::path& path) {
  YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;
  const auto inf = root["inference"];
  const auto in = root["input"];
  const auto rt = root["runtime"];
  const auto ctl = root["control"];
  const std::string default_labels =
      (path.parent_path() / "coco_label.txt").string();

  if (!root["models"] || !root["models"].IsSequence() || root["models"].size() < 2 ||
      root["models"].size() > 4)
    throw std::runtime_error("models must list between 2 and 4 entries");
  for (const auto& m : root["models"]) {
    ModelEntry e;
    e.name = m["name"].as<std::string>();
    e.path = resolve(path, m["path"].as<std::string>()).string();
    e.decode_type = lower(m["decode_type"] ? m["decode_type"].as<std::string>() : "yolo26");
    e.labels_path =
        m["labels"] ? resolve(path, m["labels"].as<std::string>()).string() : default_labels;
    e.min_score = m["min_score"] ? m["min_score"].as<double>() : 0.45;
    e.nms_iou = m["nms_iou"] ? m["nms_iou"].as<double>() : 0.60;
    e.max_detections = m["max_detections"] ? m["max_detections"].as<int>() : 50;
    e.description = m["description"] ? m["description"].as<std::string>() : "";
    for (auto& c : as_str_list(m["classes"])) e.class_filter.push_back(lower(c));
    if (!fs::is_regular_file(e.path)) throw std::runtime_error("model archive not found: " + e.path);
    if (!fs::is_regular_file(e.labels_path))
      throw std::runtime_error("labels not found: " + e.labels_path);
    cfg.models.push_back(std::move(e));
  }

  std::vector<std::string> default_models;
  if (root["default_models"]) default_models = as_str_list(root["default_models"]);
  else default_models = {cfg.models.front().name};

  if (!root["streams"] || !root["streams"].IsSequence() || root["streams"].size() == 0)
    throw std::runtime_error("streams must be a non-empty list");
  for (const auto& s : root["streams"]) {
    StreamCfg sc;
    if (s.IsScalar()) {
      sc.url = s.as<std::string>();
      sc.models = default_models;
    } else {
      sc.url = s["url"].as<std::string>();
      sc.models = s["models"] ? as_str_list(s["models"])
                              : (s["model"] ? as_str_list(s["model"]) : default_models);
    }
    if (sc.models.empty()) sc.models = default_models;
    cfg.streams.push_back(std::move(sc));
  }

  auto iod = [&](const YAML::Node& n, const char* k, int d) {
    return n && n[k] ? n[k].as<int>() : d;
  };
  cfg.input_width = iod(in, "width", 1280);
  cfg.input_height = iod(in, "height", 720);
  cfg.input_fps = iod(in, "fps", 30);
  cfg.target_fps = iod(in, "target_fps", 0);
  cfg.latency_ms = iod(in, "latency_ms", 100);
  cfg.decoder_buffers = iod(in, "decoder_buffers", 4);
  cfg.decoder_input_buffers = iod(in, "decoder_input_buffers", 1);
  if (in && in["decoder_tuning"]) cfg.decoder_tuning = in["decoder_tuning"].as<std::string>();
  cfg.queue_depth = iod(inf, "queue_depth", 4);
  cfg.internal_queue_depth = iod(inf, "internal_queue_depth", 1);
  cfg.max_inflight_per_stream = iod(inf, "max_inflight_per_stream", 2);
  cfg.max_inflight_total = iod(inf, "max_inflight_total", 8);
  if (rt && rt["watchdog_stall_s"]) cfg.watchdog_stall_s = rt["watchdog_stall_s"].as<double>();
  cfg.rebuild_retries = iod(rt, "rebuild_retries", 5);
  if (rt && rt["report_hz"]) cfg.report_hz = rt["report_hz"].as<double>();
  if (rt && rt["watchdog"]) cfg.watchdog_enabled = (rt["watchdog"].as<std::string>() != "off");
  const auto cr = root["crops"];
  if (cr) {
    if (cr["enabled"]) cfg.crops_enabled = cr["enabled"].as<bool>();
    if (cr["dir"]) cfg.crops_dir = cr["dir"].as<std::string>();
    if (cr["min_conf"]) cfg.crop_min_conf = cr["min_conf"].as<double>();
    if (cr["dedup_iou"]) cfg.crop_dedup_iou = cr["dedup_iou"].as<double>();
    if (cr["dedup_window"]) cfg.crop_dedup_window = cr["dedup_window"].as<double>();
    if (cr["pad"]) cfg.crop_pad = cr["pad"].as<double>();
    if (cr["max_disk"]) cfg.crop_max_disk = cr["max_disk"].as<int>();
    if (cr["tee_width"]) cfg.crop_tee_w = cr["tee_width"].as<int>();
    if (cr["tee_height"]) cfg.crop_tee_h = cr["tee_height"].as<int>();
    if (cr["workers"]) cfg.crop_workers = cr["workers"].as<int>();
    if (cr["streams"]) for (auto& x : cr["streams"]) cfg.crop_streams.push_back(x.as<int>());
    else if (cr["max_streams"]) { int n = cr["max_streams"].as<int>();
      for (int i = 0; i < n; ++i) cfg.crop_streams.push_back(i); }
  }
  if (ctl && ctl["host"]) cfg.control_host = ctl["host"].as<std::string>();
  cfg.control_port = iod(ctl, "port", 8600);
  return cfg;
}

std::vector<std::string> load_labels(const std::string& p) {
  std::ifstream f(p);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (!line.empty()) out.push_back(line);
  }
  if (out.empty()) throw std::runtime_error("labels file empty: " + p);
  return out;
}

// ---------------------------------------------------------------- neat helpers

neat::BoxDecodeType parse_box_decode_type(const std::string& t) {
  const std::string l = lower(t);
  if (l == "yolo26" || l == "yolov26") return neat::BoxDecodeType::YoloV26;
  if (l == "yolov8") return neat::BoxDecodeType::YoloV8;
  if (l == "yolov5") return neat::BoxDecodeType::YoloV5;
  if (l == "yolov6") return neat::BoxDecodeType::YoloV6;
  if (l == "yolov7") return neat::BoxDecodeType::YoloV7;
  if (l == "yolov9") return neat::BoxDecodeType::YoloV9;
  if (l == "yolov10") return neat::BoxDecodeType::YoloV10;
  if (l == "yolox") return neat::BoxDecodeType::YoloX;
  throw std::runtime_error("unsupported decode_type: " + t);
}

std::unique_ptr<neat::Model> make_model(const ModelEntry& e, int num_classes) {
  neat::Model::Options o;
  o.preprocess.kind = neat::InputKind::Image;
  o.preprocess.enable = neat::AutoFlag::On;
  o.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  o.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  o.decode_type = parse_box_decode_type(e.decode_type);
  o.score_threshold = static_cast<float>(e.min_score);
  o.nms_iou_threshold = static_cast<float>(e.nms_iou);
  o.top_k = e.max_detections;
  if (num_classes > 0) o.num_classes = num_classes;
  o.name_suffix = "_" + e.name;  // K models in one graph: keep element names unique
  return std::make_unique<neat::Model>(e.path, o);
}

neat::nodes::groups::RtspDecodedInputOptions make_source_options(const AppConfig& cfg,
                                                                 const std::string& url) {
  neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = true;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = neat::FormatTag::NV12;
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.decoder_next_element = "CVU";
  opt.auto_caps_from_stream = false;
  opt.num_buffers = cfg.decoder_buffers;
  opt.output_caps.enable = true;
  opt.output_caps.format = neat::FormatTag::NV12;
  opt.output_caps.memory = neat::CapsMemory::Any;
  opt.fallback_h264_width = cfg.input_width;
  opt.fallback_h264_height = cfg.input_height;
  opt.fallback_h264_fps = cfg.input_fps;
  opt.output_caps.width = cfg.input_width;
  opt.output_caps.height = cfg.input_height;
  opt.output_caps.fps = cfg.input_fps;
  return opt;
}

neat::Graph make_rtsp_h264_input(const neat::nodes::groups::RtspDecodedInputOptions& opt) {
  neat::nodes::groups::RtspEncodedInputOptions e;
  e.url = opt.url;
  e.codec = neat::nodes::groups::RtspCodec::H264;
  e.latency_ms = opt.latency_ms;
  e.tcp = opt.tcp;
  e.drop_on_latency = opt.drop_on_latency;
  e.buffer_mode = opt.buffer_mode;
  e.insert_queue = opt.insert_queue;
  e.sync_mode = opt.sync_mode;
  e.h264_payload_type = opt.payload_type;
  e.h264_parse_config_interval = opt.h264_parse_config_interval;
  e.h264_fps = opt.h264_fps;
  e.h264_width = opt.h264_width;
  e.h264_height = opt.h264_height;
  e.auto_caps_from_stream = opt.auto_caps_from_stream;
  e.fallback_h264_fps = opt.fallback_h264_fps;
  e.fallback_h264_width = opt.fallback_h264_width;
  e.fallback_h264_height = opt.fallback_h264_height;
  return neat::nodes::groups::RtspEncodedInput(e);
}

neat::Graph make_h264_decoder(const AppConfig& cfg,
                              const neat::nodes::groups::RtspDecodedInputOptions& opt) {
  neat::Graph g("h264_decoder");
  neat::SimaDecodeOptions decode;
  decode.type = neat::SimaDecodeType::H264;
  decode.sima_allocator_type = opt.sima_allocator_type;
  decode.out_format = neat::FormatTag::NV12;
  decode.decoder_name = opt.decoder_name;
  decode.raw_output = opt.decoder_raw_output;
  decode.next_element = opt.decoder_next_element;
  decode.dec_width = opt.fallback_h264_width;
  decode.dec_height = opt.fallback_h264_height;
  decode.dec_fps = opt.fallback_h264_fps;
  decode.num_buffers = cfg.decoder_buffers;
  decode.input_buffers = cfg.decoder_input_buffers;
  decode.decoder_tuning = cfg.decoder_tuning;
  decode.memory_opt =
      cfg.decoder_tuning == "low-memory" || cfg.decoder_tuning == "throughput-low-latency";
  g.add(neat::nodes::SimaDecode(std::move(decode)));
  g.add(neat::nodes::CapsRaw("NV12", opt.output_caps.width, opt.output_caps.height,
                             opt.output_caps.fps, opt.output_caps.memory));
  if (cfg.target_fps > 0 && cfg.target_fps < opt.output_caps.fps) {
    g.add(neat::nodes::VideoRate());
    g.add(neat::nodes::CapsRaw("NV12", -1, -1, cfg.target_fps, opt.output_caps.memory));
  }
  g.add(neat::nodes::Output("detector_frame"));
  return g;
}

neat::Graph build_shared_detector(const AppConfig& cfg, neat::Model& model,
                                  const std::string& name) {
  neat::Graph input_graph;
  auto in_opts = model.input_appsrc_options(false);
  const int buffers = std::max(1, in_opts.pool_max_buffers);
  in_opts.max_bytes = static_cast<std::uint64_t>(cfg.input_width) * cfg.input_height * 3U / 2U *
                      static_cast<std::uint64_t>(buffers);
  in_opts.pool_max_buffers = buffers;
  in_opts.block = true;
  input_graph.add(neat::nodes::Input("detector_frame", in_opts));

  neat::Graph out;
  out.add(neat::nodes::Output("det_" + name, neat::OutputOptions::EveryFrame(cfg.queue_depth)));

  neat::Graph model_graph = model.graph();
  neat::Graph det;
  det.connect(input_graph, model_graph);
  det.connect(model_graph, out);
  return det;
}

neat::RunOptions realtime_options(int queue_depth) {
  neat::RunOptions o;
  o.preset = neat::RunPreset::Realtime;
  o.queue_depth = queue_depth;
  o.overflow_policy = neat::OverflowPolicy::KeepLatest;
  o.output_memory = neat::OutputMemory::ZeroCopy;
  return o;
}

neat::GraphLinkOptions fanin_link(const AppConfig& cfg, const std::string& stream_id) {
  neat::GraphLinkOptions l;
  l.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  l.queue_depth = cfg.queue_depth;
  l.stream_id = stream_id;
  l.max_inflight_per_stream = cfg.max_inflight_per_stream;
  l.max_inflight_total = cfg.max_inflight_total;
  return l;
}

neat::GraphOptions graph_options(int internal_queue_depth) {
  neat::GraphOptions o;
  o.advanced_execution.internal_queue_depth = internal_queue_depth;
  o.advanced_execution.inference_async = true;
  return o;
}

// -------- bbox payload extraction (from the high-density reference) --------

const neat::Tensor* find_bbox_tensor(const neat::Sample& sample, std::string& err) {
  if (sample.kind == neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields)
      if (const auto* t = find_bbox_tensor(field, err)) return t;
    err = "bundle missing BBOX field";
    return nullptr;
  }
  const neat::Tensor* tensor = nullptr;
  if (sample.kind == neat::SampleKind::Tensor && sample.tensor.has_value())
    tensor = &*sample.tensor;
  else if (sample.kind == neat::SampleKind::TensorSet && !sample.tensors.empty())
    tensor = &sample.tensors.front();
  else {
    err = "capture_expected_tensor";
    return nullptr;
  }
  std::string fmt = sample.payload_tag;
  if (fmt.empty() && !sample.format.empty()) fmt = sample.format;
  if (fmt.empty() && tensor->semantic.tess.has_value()) fmt = tensor->semantic.tess->format;
  const std::string fu = objdet::upper_ascii_copy(fmt);
  if (!fu.empty() && fu != "BBOX") {
    err = "capture_expected_bbox format=" + fu;
    return nullptr;
  }
  return tensor;
}

bool map_bbox_payload(const neat::Sample& sample, std::vector<std::uint8_t>& out, std::string& err) {
  const auto* tensor = find_bbox_tensor(sample, err);
  if (!tensor) return false;
  try {
    out = tensor->copy_payload_bytes();
  } catch (const std::exception& ex) {
    err = std::string("capture_payload_failed err=") + ex.what();
    return false;
  }
  if (out.empty()) { err = "capture_empty_payload"; return false; }
  return true;
}

int stream_index_from_sample(const neat::Sample& sample, int count) {
  const std::string prefix = "stream";
  if (sample.stream_id.rfind(prefix, 0) != 0) {
    if (count == 1) return 0;
    throw std::runtime_error("sample missing stream id: " + sample.stream_id);
  }
  const std::string suffix = sample.stream_id.substr(prefix.size());
  int idx = std::stoi(suffix);
  if (idx < 0 || idx >= count) throw std::runtime_error("stream id out of range: " + sample.stream_id);
  return idx;
}

// ---------------------------------------------------------------- app

struct StreamState {
  int index = 0;
  std::string url;
  std::vector<std::string> models;
  int frame_w = 0, frame_h = 0;   // detector input coords (box space)
  int tee_w = 0, tee_h = 0;       // crop-tee frame dims (may be downscaled)
  std::uint64_t processed = 0;
  std::map<std::string, std::vector<objdet::Box>> latest;  // model -> boxes
  std::string last_json = "{}";
  double win_start = 0.0;
  int win_count = 0;
  double last_fps = 0.0;
  std::map<std::string, double> last_record;  // model -> ts
  std::mutex mtx;  // guards last_json / fps for the HTTP thread
  // exact-crop support: ring of recent decoded frames keyed by pts (same
  // decoder as the detector, so pts matches the detection exactly)
  struct FrameEntry { std::int64_t pts_ns; std::vector<std::uint8_t> nv12; int w; int h; };
  std::deque<FrameEntry> frames;
  std::mutex frame_mtx;
  std::deque<std::tuple<double, std::string, std::array<float,4>>> saved_boxes;  // dedup
  std::mutex crop_mtx;  // serialize crop processing (dedup state) per stream
};

// ---- crop store (shared) ----
struct CropMeta { int cam; std::string label, model, path; double conf, t; int w, h; };
std::deque<CropMeta> g_recent;              // newest-first metadata
std::mutex g_recent_mtx;
std::deque<std::string> g_crop_files;       // FIFO for disk cap
std::map<std::string, std::uint64_t> g_counts;
std::atomic<std::uint64_t> g_crop_seq{0};

float iou4(const std::array<float,4>& a, const std::array<float,4>& b) {
  float x0 = std::max(a[0], b[0]), y0 = std::max(a[1], b[1]);
  float x1 = std::min(a[0]+a[2], b[0]+b[2]), y1 = std::min(a[1]+a[3], b[1]+b[3]);
  float iw = std::max(0.f, x1-x0), ih = std::max(0.f, y1-y0), inter = iw*ih;
  float u = a[2]*a[3] + b[2]*b[3] - inter;
  return u > 0 ? inter/u : 0.f;
}

class App {
 public:
  explicit App(AppConfig cfg) : cfg_(std::move(cfg)) {
    for (std::size_t i = 0; i < cfg_.models.size(); ++i) by_name_[cfg_.models[i].name] = i;
    for (std::size_t i = 0; i < cfg_.streams.size(); ++i) {
      auto s = std::make_unique<StreamState>();
      s->index = static_cast<int>(i);
      s->url = cfg_.streams[i].url;
      s->models = cfg_.streams[i].models;
      s->frame_w = cfg_.input_width;
      s->frame_h = cfg_.input_height;
      streams_.push_back(std::move(s));
    }
    report_interval_ = 1.0 / std::max(1.0, cfg_.report_hz);
    last_progress_ts_ = now_s();
  }

  const AppConfig& cfg() const { return cfg_; }
  std::vector<std::unique_ptr<StreamState>>& streams() { return streams_; }

  void load_models() {
    for (const auto& e : cfg_.models) {
      labels_[e.name] = load_labels(e.labels_path);
      const double t0 = now_s();
      registry_[e.name] = make_model(e, static_cast<int>(labels_[e.name].size()));
      std::cout << "[model] loaded '" << e.name << "' (" << e.decode_type << ", "
                << labels_[e.name].size() << " classes) in " << (now_s() - t0) << "s\n";
    }
  }

  std::vector<std::string> models_in_use() {
    std::vector<std::string> used;
    for (auto& s : streams_)
      for (auto& m : s->models)
        if (std::find(used.begin(), used.end(), m) == used.end()) used.push_back(m);
    return used;
  }

  void rebuild() {
    std::lock_guard<std::mutex> lk(run_mtx_);
    const bool had = run_alive_;
    if (run_alive_) {
      try { run_.close(); } catch (const std::exception& e) {
        std::cerr << "[graph] close: " << e.what() << "\n";
      }
    }
    run_alive_ = false;
    graph_.reset();
    // Let the old pipeline's decoders release their admission leases before we
    // request new ones. At high stream counts this must be generous or the
    // decoder daemon reports "insufficient decoder memory" on the rebuild.
    if (had) {
      const int settle = std::max<int>(3, static_cast<int>(streams_.size()) / 6);
      std::this_thread::sleep_for(std::chrono::seconds(settle));
    }

    active_models_ = models_in_use();
    model_streams_.clear();
    for (const auto& name : active_models_) {
      std::vector<int> idxs;
      for (auto& s : streams_)
        if (std::find(s->models.begin(), s->models.end(), name) != s->models.end())
          idxs.push_back(s->index);
      model_streams_[name] = idxs;
    }

    for (int attempt = 1; g_stop.load() == 0; ++attempt) {
      try {
        graph_.emplace(graph_options(cfg_.internal_queue_depth));
        std::map<std::string, neat::Graph> detectors;
        for (const auto& name : active_models_)
          detectors.emplace(name, build_shared_detector(cfg_, *registry_[name], name));
        for (auto& s : streams_) {
          s->latest.clear();
          auto src = make_source_options(cfg_, s->url);
          auto rtsp = make_rtsp_h264_input(src);
          auto decoder = make_h264_decoder(cfg_, src);
          graph_->connect(rtsp, decoder);
          for (const auto& name : s->models)
            graph_->connect(decoder, detectors.at(name), fanin_link(cfg_, "stream" + std::to_string(s->index)));
          if (crop_on(s->index)) {
            // Tee the SAME decoded frames to a per-stream output so we can crop
            // the exact frame each detection was computed on (matched by pts).
            // At high stream counts the full-res tee exhausts CMA, so optionally
            // downscale the teed frame (boxes are scaled to match when cropping).
            neat::Graph fout;
            if (cfg_.crop_tee_w > 0 && cfg_.crop_tee_h > 0) {
              fout.add(neat::nodes::VideoScale());
              fout.add(neat::nodes::CapsRaw("NV12", cfg_.crop_tee_w, cfg_.crop_tee_h,
                                            -1, neat::CapsMemory::Any));
              s->tee_w = cfg_.crop_tee_w;
              s->tee_h = cfg_.crop_tee_h;
            } else {
              s->tee_w = cfg_.input_width;
              s->tee_h = cfg_.input_height;
            }
            // Non-blocking tee: drop old frames on overflow instead of blocking.
            // A blocking tee (drop=false) backpressures the decoder and stalls
            // the whole stream -- including detection -- when drain_frames can't
            // keep up across many streams. A few buffers keep enough recent
            // frames for exact pts-matching without ever stalling the decoder.
            neat::OutputOptions topt;
            topt.max_buffers = 6;
            topt.drop = true;
            fout.add(neat::nodes::Output("frame" + std::to_string(s->index), topt));
            graph_->connect(decoder, fout);
            s->frames.clear();
          }
        }
        run_ = graph_->build(realtime_options(cfg_.queue_depth));
        run_alive_ = true;
        ++generation_;
        last_progress_ts_ = now_s();
        std::cout << "[graph] built gen=" << generation_ << " shared_models=" << active_models_.size()
                  << " streams=" << streams_.size() << "\n";
        return;
      } catch (const std::exception& e) {
        std::cerr << "[graph] build attempt " << attempt << " failed: " << e.what() << "\n";
        run_alive_ = false;
        graph_.reset();
        if (attempt >= std::max(1, cfg_.rebuild_retries)) {
          std::cerr << "[graph] giving up; watchdog will retry\n";
          return;
        }
        for (int i = 0; i < 40 && g_stop.load() == 0; ++i)
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }

  // Coalesce rapid model switches into a single rebuild. Every switch just
  // stamps a request; one worker debounces (waits for the flurry of clicks to
  // settle) and rebuilds ONCE. This prevents the pile-up of concurrent
  // 48-stream rebuilds that made the UI look stuck.
  void request_rebuild() {
    rebuild_req_ts_ = now_s();
    rebuild_req_ = true;
  }

  void rebuild_worker() {
    while (g_stop.load() == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(150));
      if (!rebuild_req_) continue;
      // debounce: wait until 0.6 s after the most recent switch request
      while (g_stop.load() == 0 && now_s() - rebuild_req_ts_ < 0.6)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      rebuild_req_ = false;
      try {
        rebuild();
      } catch (const std::exception& e) {
        std::cerr << "[rebuild-worker] " << e.what() << "\n";
      }
    }
  }

  bool set_stream_models(int idx, std::vector<std::string> models, std::string& msg) {
    if (idx < 0 || idx >= static_cast<int>(streams_.size())) { msg = "no such stream"; return false; }
    std::vector<std::string> clean;
    for (auto& n : models) {
      if (!by_name_.count(n)) { msg = "unknown model '" + n + "'"; return false; }
      if (std::find(clean.begin(), clean.end(), n) == clean.end()) clean.push_back(n);
    }
    if (clean.empty()) { msg = "must route to at least one model"; return false; }
    streams_[idx]->models = clean;
    request_rebuild();
    msg = "rebuilding shared graph";
    return true;
  }

  bool set_all_models(std::vector<std::string> models, std::string& msg) {
    std::vector<std::string> clean;
    for (auto& n : models) {
      if (!by_name_.count(n)) { msg = "unknown model '" + n + "'"; return false; }
      if (std::find(clean.begin(), clean.end(), n) == clean.end()) clean.push_back(n);
    }
    if (clean.empty()) { msg = "must route to at least one model"; return false; }
    for (auto& s : streams_) s->models = clean;
    request_rebuild();
    msg = "rebuilding shared graph";
    return true;
  }

  void record(StreamState& s, const std::string& model, std::vector<objdet::Box> boxes,
              std::int64_t pts_ns = -1, std::int64_t frame_id = -1) {
    // optional per-model class filter (e.g. a vehicle model = COCO subset)
    const auto& entry = cfg_.models[by_name_[model]];
    const auto& labels = labels_[model];
    if (!entry.class_filter.empty()) {
      std::vector<objdet::Box> kept;
      for (auto& b : boxes) {
        if (b.class_id >= 0 && b.class_id < static_cast<int>(labels.size())) {
          if (std::find(entry.class_filter.begin(), entry.class_filter.end(),
                        lower(labels[b.class_id])) != entry.class_filter.end())
            kept.push_back(b);
        }
      }
      boxes = std::move(kept);
    }
    const double t = now_s();
    s.processed++;
    total_processed_++;
    last_progress_ts_ = t;
    if (s.win_start == 0.0) { s.win_start = t; s.win_count = 0; }
    s.win_count++;
    if (t - s.win_start >= 2.0) {
      s.last_fps = s.win_count / (t - s.win_start);
      s.win_start = t; s.win_count = 0;
    }
    s.latest[model] = std::move(boxes);

    json objs = json::array();
    int oid = 1;
    for (const auto& mn : s.models) {
      auto it = s.latest.find(mn);
      if (it == s.latest.end()) continue;
      const auto& lbls = labels_[mn];
      for (const auto& b : it->second) {
        const int x = std::max(0, static_cast<int>(b.x1));
        const int y = std::max(0, static_cast<int>(b.y1));
        int w = std::max(0, static_cast<int>(b.x2 - b.x1));
        int h = std::max(0, static_cast<int>(b.y2 - b.y1));
        if (x + w > s.frame_w) w = s.frame_w - x;
        if (y + h > s.frame_h) h = s.frame_h - y;
        const std::string label = (b.class_id >= 0 && b.class_id < static_cast<int>(lbls.size()))
                                      ? lbls[b.class_id] : "unknown";
        objs.push_back({{"id", "obj_" + std::to_string(oid++)}, {"label", label},
                        {"model", mn}, {"confidence", b.score},
                        {"bbox", {float(x), float(y), float(std::max(0, w)), float(std::max(0, h))}}});
      }
    }
    json r = {{"type", "object-detection"}, {"stream_index", s.index},
              {"models", s.models}, {"pts_ns", pts_ns}, {"frame_id", frame_id},
              {"data", {{"objects", objs}}}};
    std::lock_guard<std::mutex> lk(s.mtx);
    s.last_json = r.dump();
  }

  // Pull the teed per-stream frames and keep a small ring keyed by pts, so a
  // detection can be matched to the EXACT frame it was computed on.
  void drain_frames() {
    for (auto& sp : streams_) {
      StreamState& s = *sp;
      if (!crop_on(s.index)) continue;
      const std::string fn = "frame" + std::to_string(s.index);
      for (int i = 0; i < 20; ++i) {
        neat::Sample fs; neat::PullError pe;
        const auto st = run_.pull(fn, 0, fs, &pe);
        if (st != neat::PullStatus::Ok) break;
        const neat::Tensor* t = fs.tensor.has_value() ? &*fs.tensor
                                 : (!fs.tensors.empty() ? &fs.tensors.front() : nullptr);
        if (!t) continue;
        std::vector<std::uint8_t> bytes;
        try { bytes = t->copy_payload_bytes(); } catch (const std::exception&) { continue; }
        if (bytes.empty()) continue;
        std::lock_guard<std::mutex> lk(s.frame_mtx);
        s.frames.push_back({fs.pts_ns, std::move(bytes), s.tee_w, s.tee_h});
        while (s.frames.size() > 16) s.frames.pop_front();
      }
    }
  }

  void save_crop_bytes(int cam, const std::string& label, const std::string& model,
                       float conf, int w, int h, const std::vector<uchar>& jpg) {
    const std::string sub = cfg_.crops_dir + "/cam" + std::to_string(cam);
    std::error_code ec; fs::create_directories(sub, ec);
    const std::string fname = label + "_" + std::to_string(++g_crop_seq) + ".jpg";
    const std::string fpath = sub + "/" + fname;
    { std::ofstream f(fpath, std::ios::binary); f.write((const char*)jpg.data(), jpg.size()); }
    CropMeta m{cam, label, model, "cam" + std::to_string(cam) + "/" + fname,
               conf, now_s(), w, h};
    std::lock_guard<std::mutex> lk(g_recent_mtx);
    g_recent.push_front(m);
    while (g_recent.size() > 500) g_recent.pop_back();
    g_counts[label]++;
    g_crop_files.push_back(fpath);
    while ((int)g_crop_files.size() > cfg_.crop_max_disk) {
      std::remove(g_crop_files.front().c_str()); g_crop_files.pop_front();
    }
  }

  // Crop the exact frame this detection came from (matched by pts), with dedup.
  void maybe_crop(StreamState& s, const std::string& model,
                  const std::vector<objdet::Box>& boxes, std::int64_t pts_ns) {
    std::vector<std::uint8_t> nv12; int W = 0, H = 0; bool found = false;
    {
      std::lock_guard<std::mutex> lk(s.frame_mtx);
      for (auto it = s.frames.rbegin(); it != s.frames.rend(); ++it)
        if (it->pts_ns == pts_ns) { nv12 = it->nv12; W = it->w; H = it->h; found = true; break; }
      if (!found && !s.frames.empty()) {                 // closest fallback
        auto best = s.frames.begin(); std::int64_t bd = -1;
        for (auto it = s.frames.begin(); it != s.frames.end(); ++it) {
          std::int64_t d = std::llabs(it->pts_ns - pts_ns);
          if (bd < 0 || d < bd) { bd = d; best = it; }
        }
        nv12 = best->nv12; W = best->w; H = best->h; found = true;
      }
    }
    if (!found || W == 0) return;
    cv::Mat yuv(H * 3 / 2, W, CV_8UC1, nv12.data());
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
    // Boxes are in detector-input coords; the teed frame may be downscaled.
    const double sx = (s.frame_w > 0) ? double(W) / s.frame_w : 1.0;
    const double sy = (s.frame_h > 0) ? double(H) / s.frame_h : 1.0;
    const auto& lbls = labels_[model];
    const double nowt = now_s();
    auto& dq = s.saved_boxes;
    while (!dq.empty() && nowt - std::get<0>(dq.front()) > cfg_.crop_dedup_window) dq.pop_front();
    for (const auto& b : boxes) {
      if (b.score < cfg_.crop_min_conf) continue;
      const int cid = b.class_id;
      const std::string label = (cid >= 0 && cid < (int)lbls.size()) ? lbls[cid] : "obj";
      std::array<float,4> box{float(b.x1), float(b.y1), float(b.x2 - b.x1), float(b.y2 - b.y1)};
      bool dup = false;
      for (auto& e : dq)
        if (std::get<1>(e) == label && iou4(box, std::get<2>(e)) >= cfg_.crop_dedup_iou) { dup = true; break; }
      if (dup) continue;
      dq.push_back({nowt, label, box});
      // Map the detection box into the (possibly downscaled) tee-frame pixels.
      const float bx = box[0] * sx, by = box[1] * sy, bw = box[2] * sx, bh = box[3] * sy;
      const int px = int(bw * cfg_.crop_pad), py = int(bh * cfg_.crop_pad);
      const int x0 = std::max(0, int(bx) - px), y0 = std::max(0, int(by) - py);
      const int x1 = std::min(W, int(bx + bw) + px), y1 = std::min(H, int(by + bh) + py);
      if (x1 - x0 < 8 || y1 - y0 < 8) continue;
      std::vector<uchar> jpg;
      cv::imencode(".jpg", bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0)), jpg,
                   {cv::IMWRITE_JPEG_QUALITY, 88});
      save_crop_bytes(s.index, label, model, b.score, x1 - x0, y1 - y0, jpg);
    }
  }

  void pump() {
    double last_gc = now_s();
    std::vector<std::uint8_t> payload;
    std::vector<objdet::Box> boxes;
    while (g_stop.load() == 0) {
      std::unique_lock<std::mutex> lk(run_mtx_);
      if (!run_alive_) { lk.unlock(); std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
      bool did_work = false;
      bool failed = false;
      for (const auto& name : active_models_) {
        const std::string out_name = "det_" + name;
        for (int i = 0; i < 64; ++i) {
          neat::Sample sample;
          neat::PullError perr;
          const auto st = run_.pull(out_name, 0, sample, &perr);
          if (st == neat::PullStatus::Timeout) break;
          if (st == neat::PullStatus::Closed || st == neat::PullStatus::Error) { failed = true; break; }
          if (st != neat::PullStatus::Ok) continue;
          did_work = true;
          try {
            const auto& routed = model_streams_[name];
            int idx = routed.size() == 1 ? routed[0]
                                         : stream_index_from_sample(sample, static_cast<int>(streams_.size()));
            StreamState& s = *streams_[idx];
            if (std::find(s.models.begin(), s.models.end(), name) == s.models.end()) continue;
            const double tnow = now_s();
            auto key = name;  // per (stream,model): use map keyed by model within stream
            auto lrit = s.last_record.find(name);
            if (lrit != s.last_record.end() && tnow - lrit->second < report_interval_) continue;
            s.last_record[name] = tnow;
            std::string err;
            if (!map_bbox_payload(sample, payload, err)) continue;
            boxes.clear();
            objdet::parse_boxes_strict_into(std::span<const std::uint8_t>(payload.data(), payload.size()),
                                            s.frame_w, s.frame_h, cfg_.models[by_name_[name]].max_detections,
                                            false, boxes);
            record(s, name, boxes, sample.pts_ns, sample.frame_id);
            if (crop_on(idx)) enqueue_crop(idx, name, boxes, sample.pts_ns);
          } catch (const std::exception&) { /* stale sample across a rebuild */ }
        }
        if (failed) break;
      }
      if (cfg_.crops_enabled && !failed) {
        try { drain_frames(); } catch (const std::exception&) {}
      }
      if (failed) {
        run_alive_ = false;
        last_progress_ts_ = 0.0;
        lk.unlock();
        std::cerr << "[pump] pipeline failure; run marked dead\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        continue;
      }
      const double t = now_s();
      if (t - last_gc >= 2.0) last_gc = t;
      lk.unlock();
      std::this_thread::sleep_for(std::chrono::microseconds(did_work ? 500 : 3000));
    }
  }

  void watchdog() {
    if (!cfg_.watchdog_enabled) {
      std::cerr << "[watchdog] disabled by config\n";
      return;
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::uint64_t last_total = total_processed_;
    double last_check = now_s();
    double build_ts = now_s();
    std::vector<std::uint64_t> per_last(streams_.size(), 0);
    std::vector<double> per_seen(streams_.size(), now_s());
    int last_gen = generation_;
    const double floor_rate = std::max(1.0, streams_.size() * 1.5);
    while (g_stop.load() == 0) {
      std::this_thread::sleep_for(std::chrono::seconds(6));
      const double t = now_s();
      if (generation_ != last_gen) {  // fresh build: reset per-stream baselines
        last_gen = generation_;
        build_ts = t;
        for (std::size_t i = 0; i < streams_.size(); ++i) { per_last[i] = streams_[i]->processed; per_seen[i] = t; }
      }
      const std::uint64_t total = total_processed_;
      const double rate = (total - last_total) / std::max(0.001, t - last_check);
      last_total = total; last_check = t;

      // Per-stream stall: count streams that never started / went silent while
      // siblings keep going. Rebuilding the whole shared graph is expensive at
      // high stream counts, so only rebuild when a SYSTEMIC fraction stalled --
      // a couple of quiet cameras out of many are tolerated, not worth blowing
      // away every camera for.
      int stalled_count = 0;
      for (std::size_t i = 0; run_alive_ && i < streams_.size(); ++i) {
        const std::uint64_t p = streams_[i]->processed;
        if (p != per_last[i]) { per_last[i] = p; per_seen[i] = t; continue; }
        if (t - per_seen[i] > cfg_.watchdog_stall_s && t - build_ts > cfg_.watchdog_stall_s)
          ++stalled_count;
      }
      const int stall_threshold =
          std::max(2, static_cast<int>(streams_.size() / 4));  // >25% (min 2)
      const bool stream_stalled = stalled_count >= stall_threshold;
      const bool dead = !run_alive_;
      // Give a fresh build time to ramp before enforcing progress/throughput
      // floors -- at high stream counts (esp. with the software videoscale crop
      // tee) the pipeline needs tens of seconds to negotiate and fill, and
      // firing the throughput floor mid-ramp caused a rebuild storm.
      const bool past_grace = (t - build_ts > cfg_.watchdog_stall_s);
      const bool zero = run_alive_ && past_grace && (t - last_progress_ts_ > cfg_.watchdog_stall_s);
      const bool collapsed = run_alive_ && past_grace &&
                             (t - last_progress_ts_ <= cfg_.watchdog_stall_s) &&
                             rate < floor_rate;
      if (!(dead || zero || collapsed || stream_stalled)) continue;
      std::cerr << "[watchdog] " << (dead ? "run died" : zero ? "no progress"
                : stream_stalled ? ("streams stalled x" + std::to_string(stalled_count))
                : "throughput collapsed")
                << " (" << rate << "/s); rebuilding\n";
      rebuild();
      last_total = total_processed_; last_check = now_s();
      build_ts = now_s();
      for (std::size_t i = 0; i < streams_.size(); ++i) { per_last[i] = streams_[i]->processed; per_seen[i] = now_s(); }
      std::this_thread::sleep_for(std::chrono::seconds(6));
    }
  }

  // ---- HTTP payloads ----
  json models_json() {
    auto used = models_in_use();
    json arr = json::array();
    for (const auto& m : cfg_.models) {
      std::vector<int> strms;
      for (auto& s : streams_)
        if (std::find(s->models.begin(), s->models.end(), m.name) != s->models.end())
          strms.push_back(s->index);
      arr.push_back({{"name", m.name}, {"decode_type", m.decode_type}, {"description", m.description},
                     {"classes", labels_.count(m.name) ? labels_[m.name].size() : 0},
                     {"shared_active", std::find(used.begin(), used.end(), m.name) != used.end()},
                     {"streams", strms}});
    }
    return arr;
  }

  json streams_json() {
    json arr = json::array();
    for (auto& s : streams_) {
      std::size_t nobj = 0;
      { std::lock_guard<std::mutex> lk(s->mtx);
        try { nobj = json::parse(s->last_json).at("data").at("objects").size(); } catch (...) {} }
      arr.push_back({{"stream", s->index}, {"url", s->url}, {"models", s->models},
                     {"mode", s->models.size() > 1 ? "chain" : "route"},
                     {"fps", std::round(s->last_fps * 10) / 10}, {"processed", s->processed},
                     {"objects", nobj}});
    }
    return arr;
  }

  std::string result_json(int idx) {
    if (idx < 0 || idx >= static_cast<int>(streams_.size())) return "{}";
    std::lock_guard<std::mutex> lk(streams_[idx]->mtx);
    return streams_[idx]->last_json;
  }

  int active_count() const { return static_cast<int>(active_models_.size()); }
  int generation() const { return generation_; }
  bool alive() const { return run_alive_; }

 private:
  AppConfig cfg_;
  std::unordered_map<std::string, std::size_t> by_name_;
  std::map<std::string, std::unique_ptr<neat::Model>> registry_;
  std::map<std::string, std::vector<std::string>> labels_;
  std::vector<std::unique_ptr<StreamState>> streams_;
  std::mutex run_mtx_;
  std::optional<neat::Graph> graph_;
  neat::Run run_;
  std::atomic<bool> run_alive_{false};
  std::vector<std::string> active_models_;
  std::map<std::string, std::vector<int>> model_streams_;
  std::atomic<std::uint64_t> total_processed_{0};
  std::atomic<double> last_progress_ts_{0.0};
  std::atomic<bool> rebuild_req_{false};
  std::atomic<double> rebuild_req_ts_{0.0};
  int generation_ = 0;
  double report_interval_ = 0.1;
  // ---- decoupled crop processing ----
  // The pump only enqueues cheap box copies; worker threads do the expensive
  // frame lookup + NV12->BGR + JPEG encode + save, off the detector critical
  // path. This keeps det_ outputs drained so CMA does not back up at high
  // stream counts (the reason inline cropping froze the board at 48).
  struct CropReq { int idx; std::string model; std::vector<objdet::Box> boxes; std::int64_t pts; };
  std::deque<CropReq> crop_q_;
  std::mutex crop_q_mtx_;
  std::condition_variable crop_cv_;
  static constexpr std::size_t kCropQueueMax = 512;  // drop oldest on overflow

 public:
  // Is cropping enabled for this stream? (crops.streams / crops.max_streams subset)
  bool crop_on(int idx) const {
    if (!cfg_.crops_enabled) return false;
    if (cfg_.crop_streams.empty()) return true;
    return std::find(cfg_.crop_streams.begin(), cfg_.crop_streams.end(), idx) !=
           cfg_.crop_streams.end();
  }

  void enqueue_crop(int idx, const std::string& model,
                    const std::vector<objdet::Box>& boxes, std::int64_t pts) {
    {
      std::lock_guard<std::mutex> lk(crop_q_mtx_);
      if (crop_q_.size() >= kCropQueueMax) crop_q_.pop_front();  // shed load, never block pump
      crop_q_.push_back({idx, model, boxes, pts});
    }
    crop_cv_.notify_one();
  }

  void notify_crop_workers() { crop_cv_.notify_all(); }

  void crop_worker() {
    while (g_stop.load() == 0) {
      CropReq req;
      {
        std::unique_lock<std::mutex> lk(crop_q_mtx_);
        crop_cv_.wait_for(lk, std::chrono::milliseconds(200),
                          [&] { return g_stop.load() != 0 || !crop_q_.empty(); });
        if (g_stop.load() != 0) return;
        if (crop_q_.empty()) continue;
        req = std::move(crop_q_.front());
        crop_q_.pop_front();
      }
      if (req.idx < 0 || req.idx >= static_cast<int>(streams_.size())) continue;
      StreamState& s = *streams_[req.idx];
      std::lock_guard<std::mutex> lk(s.crop_mtx);  // dedup state is per-stream
      try { maybe_crop(s, req.model, req.boxes, req.pts); }
      catch (const std::exception&) {}
    }
  }

 private:
};

const char* kUiHtml = R"HTML(<!doctype html><html><head><meta charset="utf-8">
<title>Shared Multi-Model Detector</title><style>
body{font-family:system-ui,sans-serif;margin:24px;background:#0f1115;color:#e6e6e6}
h1{font-size:20px}table{border-collapse:collapse;width:100%;background:#171a21}
th,td{border:1px solid #2a2f3a;padding:6px 10px;font-size:14px;text-align:left}
th{background:#1f242e}.box{color:#9fe}button{font-size:13px;margin:1px}.mode{font-size:12px;color:#9aa}
</style></head><body>
<h1>SiMa Modalix &mdash; Shared Multi-Model Detector (C++)</h1>
<p class="mode">Models are loaded once as shared detectors; every camera fans into the model(s)
it routes to. Pick one (route) or several (chain) per camera; a change rebuilds the shared graph.</p>
<p>All cameras: <span id="allbtns"></span></p>
<table id="tbl"><thead><tr><th>#</th><th>Source</th><th>Models</th><th>Mode</th><th>FPS</th><th>Frames</th><th>Detections</th></tr></thead><tbody></tbody></table>
<script>
let models=[];
async function loadModels(){models=await (await fetch('api/models')).json();
 document.getElementById('allbtns').innerHTML=models.map(m=>`<button onclick="setAll('${m.name}')">${m.name}</button>`).join('');}
async function setModels(i,l){await fetch(`api/streams/${i}/models`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({models:l})});refresh();}
function toggle(i,name,cur){let l=cur.includes(name)?cur.filter(x=>x!=name):cur.concat([name]);if(l.length==0)l=[name];setModels(i,l);}
async function setAll(n){await fetch('api/models_all',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({models:[n]})});}
async function refresh(){try{
 const streams=await (await fetch('api/streams')).json();
 const results=await (await fetch('api/results')).json();
 document.querySelector('#tbl tbody').innerHTML=streams.map(s=>{const r=results[s.stream]||{};const objs=(r.data&&r.data.objects)||[];
  const summ=objs.slice(0,8).map(o=>`<span class="box">${o.label}</span>`).join(' ');
  const btns=models.map(m=>`<button style="opacity:${s.models.includes(m.name)?1:.4}" onclick='toggle(${s.stream},"${m.name}",${JSON.stringify(s.models)})'>${m.name}</button>`).join(' ');
  return `<tr><td>${s.stream}</td><td><small>${s.url.split('/').pop()}</small></td><td>${btns}</td><td class="mode">${s.mode}</td><td>${s.fps}</td><td>${s.processed}</td><td>${objs.length} ${summ}</td></tr>`;}).join('');
 }catch(e){}}
loadModels().then(refresh);setInterval(refresh,1000);
</script></body></html>)HTML";

}  // namespace

int main(int argc, char** argv) {
  fs::path config = argc > 1 ? fs::path(argv[1]) : fs::path("demo.yaml");
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--config" && i + 1 < argc) config = argv[++i];
  }
  AppConfig cfg;
  try {
    cfg = load_config(config);
  } catch (const std::exception& e) {
    std::cerr << "[ERR] config: " << e.what() << "\n";
    return 2;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  App app(std::move(cfg));

  httplib::Server srv;
  srv.Get("/", [&](const httplib::Request&, httplib::Response& r) {
    // Serve viewer.html from the config directory if present (lets the web UI
    // be iterated without recompiling); otherwise the embedded page.
    const fs::path viewer = config.parent_path() / "viewer.html";
    std::error_code ec;
    if (fs::is_regular_file(viewer, ec)) {
      std::ifstream f(viewer, std::ios::binary);
      std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      if (!body.empty()) { r.set_content(body, "text/html"); return; }
    }
    r.set_content(kUiHtml, "text/html");
  });
  srv.Get("/api/models", [&](const httplib::Request&, httplib::Response& r) { r.set_content(app.models_json().dump(), "application/json"); });
  srv.Get("/api/streams", [&](const httplib::Request&, httplib::Response& r) { r.set_content(app.streams_json().dump(), "application/json"); });
  srv.Get("/api/results", [&](const httplib::Request&, httplib::Response& r) {
    json o = json::object();
    for (auto& s : app.streams()) o[std::to_string(s->index)] = json::parse(app.result_json(s->index));
    r.set_content(o.dump(), "application/json");
  });
  srv.Get(R"(/api/results/(\d+))", [&](const httplib::Request& req, httplib::Response& r) {
    r.set_content(app.result_json(std::stoi(req.matches[1])), "application/json");
  });
  srv.Get("/api/health", [&](const httplib::Request&, httplib::Response& r) {
    r.set_content(json{{"ok", app.alive()}, {"active_models", app.active_count()}, {"generation", app.generation()}}.dump(), "application/json");
  });
  srv.Get("/api/crops", [&](const httplib::Request&, httplib::Response& r) {
    json arr = json::array(); std::uint64_t total = 0; json cnt = json::object();
    { std::lock_guard<std::mutex> lk(g_recent_mtx);
      for (auto& m : g_recent)
        arr.push_back({{"cam", m.cam}, {"label", m.label}, {"model", m.model},
                       {"conf", m.conf}, {"path", m.path}, {"w", m.w}, {"h", m.h}, {"t", m.t}});
      for (auto& c : g_counts) { cnt[c.first] = c.second; total += c.second; }
    }
    r.set_content(json{{"crops", arr}, {"counts", cnt}, {"total", total}}.dump(), "application/json");
  });
  srv.Get(R"(/crop/(.+))", [&](const httplib::Request& req, httplib::Response& r) {
    std::string rel = req.matches[1];
    if (rel.find("..") != std::string::npos) { r.status = 404; return; }
    std::ifstream f(app.cfg().crops_dir + "/" + rel, std::ios::binary);
    if (!f) { r.status = 404; return; }
    std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    r.set_content(body, "image/jpeg");
  });
  auto parse_models = [](const httplib::Request& req) {
    auto b = json::parse(req.body, nullptr, false);
    std::vector<std::string> m;
    if (b.is_object() && b.contains("models")) for (auto& x : b["models"]) m.push_back(x.get<std::string>());
    else if (b.is_object() && b.contains("model")) m.push_back(b["model"].get<std::string>());
    return m;
  };
  srv.Post("/api/models_all", [&](const httplib::Request& req, httplib::Response& r) {
    std::string msg; bool ok = app.set_all_models(parse_models(req), msg);
    r.status = ok ? 202 : 400; r.set_content(json{{"queued", ok}, {"detail", msg}}.dump(), "application/json");
  });
  srv.Post(R"(/api/streams/(\d+)/models)", [&](const httplib::Request& req, httplib::Response& r) {
    std::string msg; bool ok = app.set_stream_models(std::stoi(req.matches[1]), parse_models(req), msg);
    r.status = ok ? 202 : 400; r.set_content(json{{"queued", ok}, {"detail", msg}}.dump(), "application/json");
  });

  std::thread http_thread([&] { srv.listen(app.cfg().control_host.c_str(), app.cfg().control_port); });
  std::cout << "[control] UI + API on http://" << app.cfg().control_host << ":" << app.cfg().control_port << "/\n";

  app.load_models();
  std::thread builder([&] { app.rebuild(); });
  std::thread dog([&] { app.watchdog(); });
  std::thread rw([&] { app.rebuild_worker(); });

  // Crop worker pool: does the expensive NV12->BGR + JPEG encode off the pump
  // so detection draining (and CMA) stay healthy at high stream counts.
  std::vector<std::thread> crop_workers;
  if (app.cfg().crops_enabled) {
    const int nw = app.cfg().crop_workers > 0 ? app.cfg().crop_workers : 3;
    for (int i = 0; i < nw; ++i) crop_workers.emplace_back([&] { app.crop_worker(); });
    std::cout << "[crops] " << nw << " worker threads\n";
  }

  app.pump();

  g_stop.store(1);
  app.notify_crop_workers();
  srv.stop();
  if (http_thread.joinable()) http_thread.join();
  if (builder.joinable()) builder.join();
  if (dog.joinable()) dog.join();
  if (rw.joinable()) rw.join();
  for (auto& t : crop_workers) if (t.joinable()) t.join();
  return 0;
}
