import argparse
import csv
import multiprocessing as mp
import os
import threading
from time import sleep

from worker_proc import WorkerProc


class LoggingThd(threading.Thread):
    def __init__(self, log_fname, queue):
        super(LoggingThd, self).__init__()
        self.queue = queue
        self.csv_fh = open(log_fname, 'w', 1)
        self.csv_writer = csv.writer(self.csv_fh, lineterminator='\n')
        self.csv_writer.writerow(
            ['start_timestamp_ns', 'end_timestamp_ns', 'jct_ms',
             'max_allocated_gpu_memory_allocated_byte',
             'max_reserved_gpu_memory_byte'])

    def run(self):
        while True:
            (start_t, end_t) = self.queue.get()
            self.csv_writer.writerow([
                start_t, end_t, (end_t - start_t) / 1000000])
            # , max_alloc_mem_byte, max_rsrv_mem_byte])


def parse_args():
    parser = argparse.ArgumentParser(description="A DNN inference container.")
    parser.add_argument('--model-name', metavar="MODEL_NAME", type=str,
                        default="fasterrcnn_resnet50_fpn",
                        help="DNN model name to run")
    parser.add_argument('--model-weight', metavar="MODEL_WEIGHT",
                        default='FasterRCNN_ResNet50_FPN_Weights',
                        type=str, help="Model weight to load.")
    parser.add_argument('--device-id', metavar="DEVICE", type=int,
                        required=False, default=0, help="Device id")
    parser.add_argument('--output-path', metavar='PATH', type=str, default='.',
                        help="Path to save the output.")
    parser.add_argument('--name', metavar='NAME', type=str, default='model_A',
                        help="Name of the output.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    model_weight = args.model_weight
    device_id = args.device_id
    output_path = args.output_path
    output_file_name = args.name

    req_queue = mp.Queue()
    res_queue = mp.Queue()

    # set up log
    os.makedirs(output_path, exist_ok=True)
    csv_fname = os.path.join(output_path, output_file_name + ".csv")
    logging_thd = LoggingThd(csv_fname, res_queue)
    logging_thd.daemon = True
    logging_thd.start()

    worker_proc = WorkerProc(req_queue, res_queue, model_name, model_weight, device_id)
    worker_proc.daemon = True
    worker_proc.start()

    while True:
        req_queue.put(
            b'{"batch_size": 1, "resize_size": [720, 1280], "input_file_path": "dataset/rene/0000000099.png"}')
        sleep(1)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
