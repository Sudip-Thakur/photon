# rpicam-vid -t 0 --width 640 --height 480 --framerate 30 --codec mjpeg --nopreview --inline --listen -o tcp://0.0.0.0:8080
# rpicam-vid -t 0 --width 640 --height 480 --framerate 60 --codec h264 --nopreview --inline --listen -o tcp://0.0.0.0:8080
# rpicam-vid -t 0 --width 640 --height 480 --framerate 30 --codec h264 --nopreview --inline --profile baseline -o udp://10.42.0.1:8080
rpicam-vid -t 0 --width 640 --height 480 --framerate 30 --codec h264 --nopreview --inline --profile baseline --intra 15 --flush 1 -o udp://192.168.8.20:9000?pkt_size=1316
