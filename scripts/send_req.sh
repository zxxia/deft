#!/usr/bin/env bash

function send_low_prior_req() {
    while true; do
        curl -X POST localhost:5000 -d \
            '{"model_name": "fasterrcnn_resnet50_fpn",
              "batch_size": 1,
              "resize_size": [720, 1280],
              "priority": 0,
              "input_file_path": "dataset/rene/0000000099.png"}'
        sleep 0.2
    done
}

function send_high_prior_req() {
    while true; do
        curl -X POST localhost:5000 -d \
            '{"model_name": "resnet50",
              "batch_size": 1,
              "resize_size": [720, 1280],
              "priority": 1,
              "input_file_path": "dataset/rene/0000000099.png"}'
        sleep 1
    done
}

send_low_prior_req &
low_pid=$!
send_high_prior_req &
high_pid=$!
sleep 30
kill $low_pid
kill $high_pid
