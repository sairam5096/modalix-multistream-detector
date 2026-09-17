// overlay-detector: one RTSP camera -> SiMa HW decode -> YOLO on CVU/MLA -> boxes drawn ON THE SOM
// (OpenCV on the A65) -> SiMa HW H.264 encode -> RTP to an Insight video channel.
// C++ port of overlay_tenant.py (same graph shape, same egress contract, app-side frame/box pairing).
//
//   overlay-detector --url rtsp://... --channel N [--host IP] [--fps 5] [--width 1280] [--height 720]
//                    [--model models/yolo26n-det-int8-b1.tar.gz] [--decode yolo26|yolov8|yolov6]
//                    [--labels labels.txt] [--bitrate 2000] [--min-score 0.30] [--dec-bufs 4]
//                    [--dec-tuning low-memory] [--dec-memopt 1] [--out-q 2] [--save-frame /tmp/x.jpg]
#include "neat.h"
#include "support/object_detection/obj_detection_utils.h"

#include <nodes/groups/VideoSender.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace neat = simaai::neat;


namespace {

struct Args {
  std::string url;
  int channel = 0;
  std::string host = "127.0.0.1";
  int fps = 5, width = 1280, height = 720;
  std::string model = "models/yolo26n-det-int8-b1.tar.gz";
  std::string decode = "yolo26";
  std::string labels = "labels.txt";
  int bitrate = 2000;
  float min_score = 0.30f, nms_iou = 0.60f;
  int max_det = 50;
  int dec_bufs = 4, dec_in_bufs = 2, out_q = 2, mla_pool = 0;
  std::string dec_tuning = "low-memory";
  bool dec_memopt = true;
  int video_port_base = 29336;
  double stall_exit_s = 60.0;
  std::string save_frame;
  int max_lag = 3;
  std::string encoder = "hw"; // hw = SiMa H.264 encoder (EV74/CMA buffers), sw = CPU x264/openh264 (no CMA)
  int push_pool = 4;        // reusable EV74 encoder-input tensors (0 = allocate one per frame)
  int alloc_retry_ms = 1000; // how long to retry a failed DMA-BUF allocation before dropping the frame
};

Args parse(int argc, char** argv) {
  Args a;
  auto next = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--url") a.url = next(i);
    else if (k == "--channel") a.channel = std::stoi(next(i));
    else if (k == "--host") a.host = next(i);
    else if (k == "--fps") a.fps = std::stoi(next(i));
    else if (k == "--width") a.width = std::stoi(next(i));
    else if (k == "--height") a.height = std::stoi(next(i));
    else if (k == "--model") a.model = next(i);
    else if (k == "--decode") a.decode = next(i);
    else if (k == "--labels") a.labels = next(i);
    else if (k == "--bitrate") a.bitrate = std::stoi(next(i));
    else if (k == "--min-score") a.min_score = std::stof(next(i));
    else if (k == "--nms-iou") a.nms_iou = std::stof(next(i));
    else if (k == "--max-det") a.max_det = std::stoi(next(i));
    else if (k == "--dec-bufs") a.dec_bufs = std::stoi(next(i));
    else if (k == "--dec-in-bufs") a.dec_in_bufs = std::stoi(next(i));
    else if (k == "--dec-tuning") a.dec_tuning = next(i);
    else if (k == "--dec-memopt") a.dec_memopt = next(i) != "0";
    else if (k == "--out-q") a.out_q = std::stoi(next(i));
    else if (k == "--mla-pool") a.mla_pool = std::stoi(next(i));
    else if (k == "--video-port-base") a.video_port_base = std::stoi(next(i));
    else if (k == "--stall-exit-s") a.stall_exit_s = std::stod(next(i));
    else if (k == "--save-frame") a.save_frame = next(i);
    else if (k == "--encoder") a.encoder = next(i);
    else if (k == "--push-pool") a.push_pool = std::stoi(next(i));
    else if (k == "--alloc-retry-ms") a.alloc_retry_ms = std::stoi(next(i));
    else if (k == "--max-lag") a.max_lag = std::stoi(next(i));
    else throw std::runtime_error("unknown option " + k);
  }
  if (a.url.empty()) throw std::runtime_error("--url is required");
  if (const char* e = std::getenv("VIDEO_PORT_BASE")) a.video_port_base = std::atoi(e);
  return a;
}

neat::BoxDecodeType decode_type(const std::string& s) {
  if (s == "yolo26") return neat::BoxDecodeType::YoloV26;
  if (s == "yolov8") return neat::BoxDecodeType::YoloV8;
  if (s == "yolov6") return neat::BoxDecodeType::YoloV6;
  if (s == "yolov5") return neat::BoxDecodeType::YoloV5;
  if (s == "yolov10") return neat::BoxDecodeType::YoloV10;
  throw std::runtime_error("--decode must be yolo26|yolov8|yolov6|yolov5|yolov10");
}

std::vector<std::string> load_labels(const std::string& path) {
  std::vector<std::string> out;
  std::ifstream in(path);
  for (std::string line; std::getline(in, line);) out.push_back(line);
  return out;
}

double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

const neat::Tensor* first_tensor(const neat::Sample& s) {
  if (s.kind == neat::SampleKind::Tensor && s.tensor) return &*s.tensor;
  if (s.kind == neat::SampleKind::TensorSet && !s.tensors.empty()) return &s.tensors.front();
  for (const auto& f : s.fields)
    if (const auto* t = first_tensor(f)) return t;
  return nullptr;
}

const cv::Scalar kPalette[] = {{0, 220, 0}, {0, 165, 255}, {255, 128, 0}, {0, 0, 230}, {200, 0, 200}, {0, 200, 200}};

void draw(cv::Mat& bgr, const std::vector<objdet::Box>& boxes, const std::vector<std::string>& labels,
          int channel, int W, int H) {
  for (const auto& b : boxes) {
    const int x1 = std::max(0, int(b.x1)), y1 = std::max(0, int(b.y1));
    const int x2 = std::min(W - 1, int(b.x2)), y2 = std::min(H - 1, int(b.y2));
    if (x2 <= x1 || y2 <= y1) continue;
    const cv::Scalar color = kPalette[(b.class_id >= 0 ? b.class_id : 0) % 6];
    cv::rectangle(bgr, {x1, y1}, {x2, y2}, color, 2);
    const std::string name = (b.class_id >= 0 && b.class_id < int(labels.size())) ? labels[b.class_id] : std::to_string(b.class_id);
    char txt[96];
    std::snprintf(txt, sizeof txt, "%s %.2f", name.c_str(), b.score);
    int base = 0;
    const cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    const int ty = (y1 - ts.height - 6 > 0) ? y1 - 4 : y1 + ts.height + 4;
    cv::rectangle(bgr, {x1, ty - ts.height - 3}, {x1 + ts.width + 4, ty + 2}, color, cv::FILLED);
    cv::putText(bgr, txt, {x1 + 2, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
  }
  char foot[64];
  std::snprintf(foot, sizeof foot, "SOM overlay (C++)  ch%d  %zu obj", channel, boxes.size());
  cv::putText(bgr, foot, {10, H - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 1, cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) try {
  const Args a = parse(argc, argv);
  const int W = a.width, H = a.height, FPS = a.fps;
  const auto labels = load_labels(a.labels);
  auto log = [&](const std::string& m) { std::cout << "[overlay-cpp ch" << a.channel << "] " << m << std::endl; };

  // ---------------- source: encoded RTSP -> SiMa decoder (lean settings)
  neat::nodes::groups::RtspEncodedInputOptions enc;
  enc.url = a.url;
  enc.codec = neat::nodes::groups::RtspCodec::H264;
  enc.tcp = true;
  enc.latency_ms = 100;
  enc.drop_on_latency = true;
  enc.insert_queue = true;
  enc.auto_caps_from_stream = false;
  enc.payload_type = 96;
  enc.fallback_h264_fps = FPS;
  enc.fallback_h264_width = W;
  enc.fallback_h264_height = H;

  neat::SimaDecodeOptions dec;
  dec.type = neat::SimaDecodeType::H264;
  dec.out_format = neat::FormatTag::NV12;
  dec.decoder_name = "decoder";
  dec.raw_output = true;
  dec.next_element = "CVU";
  dec.dec_width = W;
  dec.dec_height = H;
  dec.dec_fps = FPS;
  dec.num_buffers = a.dec_bufs;
  dec.input_buffers = a.dec_in_bufs;
  dec.decoder_tuning = a.dec_tuning;
  dec.memory_opt = a.dec_memopt;
  dec.sima_allocator_type = 2;

  neat::Graph source("source");
  source.add(neat::nodes::groups::RtspEncodedInput(enc));
  source.add(neat::nodes::SimaDecode(dec));
  source.add(neat::nodes::Output("source"));

  // ---------------- model on CVU + MLA with in-graph box decode (input_max_* left unset)
  neat::Model::Options mo;
  mo.preprocess.kind = neat::InputKind::Image;
  mo.preprocess.enable = neat::AutoFlag::On;
  mo.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  mo.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  mo.decode_type = decode_type(a.decode);
  mo.score_threshold = a.min_score;
  mo.nms_iou_threshold = a.nms_iou;
  mo.top_k = a.max_det;
  if (a.mla_pool > 0) mo.processmla.output_pool_buffers = a.mla_pool;
  mo.verbose = neat::VerboseOptions::quiet();
  neat::Model model(a.model, mo);
  log("model " + a.model + " decode " + a.decode + " labels " + std::to_string(labels.size()));

  auto branch = neat::graphs::Branch("source", {"model", "frame"});
  neat::Graph model_graph("model");
  model_graph.connect(neat::nodes::Input("model"), model);
  neat::Graph detections_graph("detections");
  detections_graph.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(a.out_q)));
  neat::Graph frame_graph("frame");
  frame_graph.add(neat::nodes::Output("frame", neat::OutputOptions::EveryFrame(a.out_q)));

  neat::Graph graph("overlay-detector");
  graph.connect(source, branch);
  graph.connect(branch, model_graph);
  graph.connect(model_graph, detections_graph);
  graph.connect(branch, frame_graph);

  neat::RunOptions ro;
  ro.preset = neat::RunPreset::Realtime;
  ro.queue_depth = 3;
  ro.overflow_policy = neat::OverflowPolicy::KeepLatest;
  ro.output_memory = neat::OutputMemory::ZeroCopy;
  log("building detector graph for " + a.url);
  neat::Run run = graph.build(ro);
  log("detector running");

  // ---------------- egress: BGR frames -> RGB EV74 tensors -> HW H.264 -> RTP -> Insight
  const bool sw_enc = (a.encoder == "sw");
  const auto push_mem = sw_enc ? neat::TensorMemory::CPU : neat::TensorMemory::EV74;
  neat::InputOptions io;
  io.payload_type = neat::PayloadType::Image;
  io.format = "BGR";
  io.width = W;
  io.height = H;
  io.depth = 3;
  io.fps_n = FPS;
  io.fps_d = 1;
  io.memory_policy = sw_enc ? neat::InputMemoryPolicy::SystemMemory : neat::InputMemoryPolicy::Ev74;
  neat::Graph egress("insight");
  egress.add(neat::nodes::Input(io));
  int video_port = a.video_port_base + a.channel;
  if (sw_enc) {
    // CPU path: no EV74/CMA buffers at all. BGR -> I420 -> x264/openh264 -> RTP -> UDP
    egress.add(neat::nodes::VideoConvert());
    egress.add(neat::nodes::H264EncodeSW(a.bitrate));
    egress.add(neat::nodes::H264Packetize());
    neat::UdpOutputOptions uo;
    uo.host = a.host;
    uo.port = video_port;
    egress.add(neat::nodes::UdpOutput(uo));
  } else {
    auto so = neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(W, H, FPS);
    so.host = a.host;
    so.channel = a.channel;
    so.video_port_base = a.video_port_base;
    so.encoder.bitrate_kbps = a.bitrate;
    video_port = so.video_port();
    egress.add(neat::nodes::groups::VideoSender(so));
  }
  cv::Mat seed(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
  neat::Run sender = egress.build(neat::TensorList{neat::Tensor::from_cv_mat(seed, neat::ImageSpec::PixelFormat::BGR, push_mem)});
  log("encoder (" + a.encoder + ") running -> Insight " + a.host + " video port " + std::to_string(video_port) + " bitrate " + std::to_string(a.bitrate) + " kbps");

  // ---------------- main loop: pair frames and detections by frame_id (tolerates dropped frames)
  std::map<int64_t, cv::Mat> pending_frames;
  std::map<int64_t, std::vector<objdet::Box>> pending_boxes;
  std::vector<objdet::Box> last_boxes, boxes;
  std::vector<std::uint8_t> payload;
  std::uint64_t frames = 0, pushed = 0, unpaired = 0, boxes_total = 0, win = 0;
  double t_conv = 0, t_draw = 0, t_push = 0;
  const double t0 = now_s();
  double last_report = t0, last_frame = t0;
  bool saved = false;
  std::uint64_t alloc_retries = 0, dropped = 0;
  int alloc_fail_logged = 0, push_err_logged = 0;

  // Encoder-input tensors. Allocating a fresh EV74 (CMA) tensor per frame fails now and then when CMA is nearly
  // full (page cache must be migrated out first), so keep a small ring of tensors and overwrite them in place.
  // If a tensor cannot be mapped, or the pool is disabled, fall back to per-frame allocation with retries.
  const auto BGRF = neat::ImageSpec::PixelFormat::BGR;
  auto alloc_with_retry = [&](const cv::Mat& m, neat::Tensor& out) -> bool {
    const double t_end = now_s() + a.alloc_retry_ms / 1000.0;
    int tries = 0;
    while (true) {
      try { out = neat::Tensor::from_cv_mat(m, BGRF, push_mem); return true; }
      catch (const std::exception& e) {
        ++tries; ++alloc_retries;
        if (now_s() >= t_end) { if (alloc_fail_logged++ < 20) log(std::string("EV74 alloc failed after ") + std::to_string(tries) + " tries: " + e.what()); return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min(5 * tries, 50)));
      }
    }
  };
  std::vector<neat::Tensor> pool;
  for (int i = 0; i < (sw_enc ? 0 : a.push_pool); ++i) {
    neat::Tensor t;
    if (!alloc_with_retry(seed, t)) break;
    pool.push_back(std::move(t));
  }
  bool pool_ok = !pool.empty();
  std::size_t pool_i = 0;
  log("encoder input pool: " + std::to_string(pool.size()) + " reusable EV74 tensors");

  auto render_and_push = [&](cv::Mat& bgr, const std::vector<objdet::Box>& bx) {
    const double b = now_s();
    draw(bgr, bx, labels, a.channel, W, H);
    if (!a.save_frame.empty() && !saved && pushed >= 40) {
      cv::imwrite(a.save_frame, bgr);
      saved = true;
      log("saved annotated frame to " + a.save_frame);
    }
    const double c = now_s();
    bool ok = false;
    try {
      neat::Tensor t;
      bool have = false;
      if (pool_ok && bgr.isContinuous()) {
        neat::Tensor& slot = pool[pool_i % pool.size()];
        const std::size_t need = static_cast<std::size_t>(W) * H * 3;
        try {
          auto m = slot.map_write();
          if (m.data && m.size_bytes >= need) { std::memcpy(m.data, bgr.data, need); have = true; }
        } catch (const std::exception& e) { log(std::string("pool map failed, using per-frame allocation: ") + e.what()); }
        if (have) { t = slot; ++pool_i; } else { pool_ok = false; }
      }
      if (!have) have = alloc_with_retry(bgr, t);
      if (have) ok = sender.push(neat::TensorList{t});
      else ++dropped;
    } catch (const std::exception& e) {
      if (push_err_logged++ < 20) log(std::string("push exception (frame dropped): ") + e.what());
      ++dropped;
    }
    const double d = now_s();
    t_draw += c - b;
    t_push += d - c;
    if (ok) ++pushed;
    else if (frames < 5 || frames % 100 == 0) log("push failed: " + sender.last_error());
  };

  while (true) {
    bool got = false;
    if (auto sd = run.pull("detections", 100)) {
      got = true;
      boxes.clear();
      std::string err;
      bool have = objdet::extract_bbox_payload(*sd, payload, err);
      if (!have) {
        // fall back to the first tensor's raw payload (what the Python tenant does)
        if (const neat::Tensor* bt = first_tensor(*sd)) {
          try { payload = bt->copy_payload_bytes(); have = !payload.empty(); } catch (const std::exception& e) { err += std::string(" | copy: ") + e.what(); }
        }
        static int warned = 0;
        if (!have && warned++ < 3) log("bbox extraction failed: " + err + " (kind=" + std::to_string(int(sd->kind)) + " tag=" + sd->payload_tag + " fmt=" + sd->format + " ntensors=" + std::to_string(sd->tensors.size()) + " fields=" + std::to_string(sd->fields.size()) + ")");
      }
      if (have) {
        try { objdet::parse_boxes_strict_into(payload, W, H, a.max_det, false, boxes); }
        catch (const std::exception& e) { static int w2 = 0; if (w2++ < 3) log(std::string("bbox parse failed: ") + e.what()); boxes.clear(); }
      }
      boxes_total += boxes.size();
      const int64_t fid = sd->frame_id;
      auto it = pending_frames.find(fid);
      if (it != pending_frames.end()) {
        render_and_push(it->second, boxes);
        last_boxes = boxes;
        pending_frames.erase(it);
      } else {
        pending_boxes[fid] = boxes;
      }
    }
    if (auto sf = run.pull("frame", 0)) {
      got = true;
      ++frames;
      ++win;
      last_frame = now_s();
      const int64_t fid = sf->frame_id;
      const neat::Tensor* ft = first_tensor(*sf);
      if (ft) {
        const double a0 = now_s();
        cv::Mat bgr = ft->to_cv_mat_copy(neat::ImageSpec::PixelFormat::BGR);
        t_conv += now_s() - a0;
        auto bit = pending_boxes.find(fid);
        if (bit != pending_boxes.end()) {
          render_and_push(bgr, bit->second);
          last_boxes = bit->second;
          pending_boxes.erase(bit);
        } else {
          pending_frames[fid] = std::move(bgr);
        }
        // frames whose detections never arrive (detector dropped them): push with the last boxes
        for (auto pit = pending_frames.begin(); pit != pending_frames.end();) {
          if (pit->first < fid - a.max_lag) {
            render_and_push(pit->second, last_boxes);
            ++unpaired;
            pit = pending_frames.erase(pit);
          } else {
            ++pit;
          }
        }
        for (auto bit2 = pending_boxes.begin(); bit2 != pending_boxes.end();) {
          if (bit2->first < fid - 10 * a.max_lag) bit2 = pending_boxes.erase(bit2);
          else ++bit2;
        }
      }
    }
    if (!got) {
      if (!run.running()) {
        log("detector stopped: " + run.last_error());
        return 2;
      }
      if (now_s() - last_frame > a.stall_exit_s) {
        log("no frames for " + std::to_string(int(a.stall_exit_s)) + " s; exiting for relaunch");
        return 2;
      }
    }
    const double now = now_s();
    if (now - last_report >= 10) {
      char buf[384];
      std::snprintf(buf, sizeof buf,
                    "frames=%llu pushed=%llu unpaired=%llu fps=%.2f avg_boxes=%.2f nv12->bgr=%.1fms draw+rgb=%.1fms push=%.1fms pend=%zu alloc_retries=%llu dropped=%llu",
                    (unsigned long long)frames, (unsigned long long)pushed, (unsigned long long)unpaired,
                    win / (now - last_report), frames ? double(boxes_total) / frames : 0.0,
                    1000 * t_conv / std::max<std::uint64_t>(1, frames), 1000 * t_draw / std::max<std::uint64_t>(1, pushed),
                    1000 * t_push / std::max<std::uint64_t>(1, pushed), pending_frames.size(),
                    (unsigned long long)alloc_retries, (unsigned long long)dropped);
      log(buf);
      last_report = now;
      win = 0;
    }
  }
} catch (const std::exception& e) {
  std::cerr << "[ERR] " << e.what() << std::endl;
  return 1;
}
