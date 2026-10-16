#!/usr/bin/env bash
# Downloads the vehicle detector used by the dashcam lead-vehicle analysis (backend/dashcam/lead_vehicle.py)
# and checks its SHA-256. YOLOX-S, OpenCV Model Zoo (https://github.com/opencv/opencv_zoo,
# models/object_detection_yolox), Apache-2.0; licence text: licenses/opencv_zoo_yolox-Apache-2.0.txt.
# Without the file, CARAI runs without a detector and says so in the case record.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
F="$D/object_detection_yolox_2022nov.onnx"
URL=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx
SHA=c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063
if [ ! -f "$F" ]; then
  python3 -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$URL" "$F"
fi
echo "$SHA  $F" | sha256sum -c -
