while true; do
    curl -X POST localhost:5000 -d \
        '{"model_name": "fasterrcnn_resnet50_fpn",
          "batch_size": 1,
          "resize_size": [720, 1280],
          "priority": 0,
          "input_file_path": "/home/zxxia/gpu_sched_new/gpu-sched-exp/data-set/rene/0000000099.png"}'
    curl -X POST localhost:5000 -d \
        '{"model_name": "resnet50",
          "batch_size": 1,
          "resize_size": [720, 1280],
          "priority": 1,
          "input_file_path": "/home/zxxia/gpu_sched_new/gpu-sched-exp/data-set/rene/0000000099.png"}'
    # break
    # sleep 1
done
