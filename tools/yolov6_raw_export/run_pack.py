"""Minimal PyNeat runner for a raw-head YOLOv6 pack on Neat 0.6.0.

  python run_pack.py <pack_mpk.tar.gz> <image.jpg> <num_classes> [out.jpg]

Prints decoded boxes in original-image pixels and flags any frame-sized box at the origin
(the 0.6.0 symptom). Needs: pyneat, numpy, opencv-python (cv2) for reading/annotating the image.
"""
import sys, time
import numpy as np, cv2, pyneat

pack, image, nc = sys.argv[1], sys.argv[2], int(sys.argv[3])
out_jpg = sys.argv[4] if len(sys.argv) > 4 else None
img = cv2.imread(image); H, W = img.shape[:2]

mo = pyneat.ModelOptions()
# --- the YOLOv6 / 0.6.0 part: all of these are required for a raw-head pack ---
mo.decode_type = pyneat.BoxDecodeType.YoloV6
mo.decode_type_option = pyneat.BoxDecodeTypeOption.GroupedByRoleLogit
mo.num_classes = nc
# --- post-processing: PyNeat defaults are 0, and top_k=0 fails the graph build ("top_k is unresolved") ---
mo.score_threshold = 0.30
mo.nms_iou_threshold = 0.60
mo.top_k = 100
# --- preprocessing: let the runtime resize/convert; model was trained on RGB, 0..1 (COCO_YOLO preset) ---
mo.preprocess.kind = pyneat.InputKind.Image
mo.preprocess.enable = pyneat.AutoFlag.On
mo.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR   # cv2 gives BGR
mo.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO

t0 = time.time(); m = pyneat.Model(pack, mo); print(f"[load] {time.time()-t0:.2f}s")
out = m.run([pyneat.Tensor.from_numpy(img, image_format=pyneat.PixelFormat.BGR)])
dets = pyneat.decode_bbox(out, clamp_to=(W, H))            # -> [N,6]: x1,y1,x2,y2,score,cls in image pixels
arr = np.asarray(dets[0].to_numpy()).reshape(-1, 6)
big = sum(1 for x1, y1, x2, y2, s, c in arr if x1 <= 1 and y1 <= 1 and (x2-x1) > 0.8*W and (y2-y1) > 0.8*H)
print(f"{len(arr)} boxes, {big} frame-sized at origin  ({'OK' if big == 0 else 'DECODE BROKEN'})")
for x1, y1, x2, y2, s, c in arr[:10]:
    print(f"  cls={int(c)} score={s:.2f} box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
if out_jpg:
    for x1, y1, x2, y2, s, c in arr:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{int(c)}:{s:.2f}", (int(x1), max(12, int(y1)-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out_jpg, img); print("wrote", out_jpg)
