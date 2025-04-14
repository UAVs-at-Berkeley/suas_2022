# ffmpegcv
import ffmpegcv
cap = ffmpegcv.VideoCaptureStream(stream_url)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass

# ffmpegcv, IP Camera Low-latency
stream_url = 'rtsp://192.168.144.25:8554/main.264'
cap = ffmpegcv.VideoCaptureStreamRT(stream_url)  # Low latency & recent buffered
cap = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, stream_url) #no buffer
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass