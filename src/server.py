import argparse
import csv
import multiprocessing as mp
import os
import requests
import threading
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from time import perf_counter_ns

from worker_proc import WorkerProc


class LoggingThd(threading.Thread):
    def __init__(self, log_fname, queue, ctlr_ip=None, ctlr_port=None):
        super(LoggingThd, self).__init__()
        self.queue = queue
        self.ctlr_ip = ctlr_ip
        self.ctlr_port = ctlr_port
        self.csv_fh = open(log_fname, 'w', 1)
        self.csv_writer = csv.writer(self.csv_fh, lineterminator='\n')
        self.csv_writer.writerow(
            ['inf_start_timestamp_ns', 'inf_end_timestamp_ns', 'jct_ms',
             'req_recv_timestamp_ns', 'log_timestamp_ns'])

    def run(self):
        while True:
            (start_t, end_t, req_recv_t) = self.queue.get()
            log_t = perf_counter_ns()
            self.csv_writer.writerow([
                start_t, end_t, (end_t - start_t) / 1000000, req_recv_t, log_t])
            # , max_alloc_mem_byte, max_rsrv_mem_byte])
            if self.ctlr_ip and self.ctlr_port:
                _ = requests.post(
                    'http://{}:{}/job_finish'.format(
                        self.ctlr_ip, self.ctlr_port))

class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, queue: mp.Queue, *args, **kwargs):
        self.queue = queue
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        self.queue.put(post_data)

        # alternative way: send the reponse until the inference is done
        self.send_response(HTTPStatus.ACCEPTED)
        self.send_header('Content-Type', 'text')
        self.end_headers()


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
    parser.add_argument('--ip', metavar='IP', type=str, default='localhost',
                        help="IP address of the container.")
    parser.add_argument('--port', metavar='PORT', type=int, default=12345,
                        help="Port the container listens to.")
    parser.add_argument('--ctlr-ip', metavar='CONTROLLER_IP', type=str,
                        default=None, help="IP address of the controller.")
    parser.add_argument('--ctlr-port', metavar='CONTROLLER_PORT', type=int,
                        default=None, help="Port the controller listens to.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    model_weight = args.model_weight
    device_id = args.device_id
    output_path = args.output_path
    output_file_name = args.name

    ip = args.ip
    port = args.port
    server_address = (ip, port)
    req_queue = mp.Queue()
    res_queue = mp.Queue()
    handler = partial(RequestHandler, req_queue)
    httpd = HTTPServer(server_address, handler)

    # set up log
    os.makedirs(output_path, exist_ok=True)
    csv_fname = os.path.join(output_path, output_file_name + ".csv")
    logging_thd = LoggingThd(csv_fname, res_queue, args.ctlr_ip, args.ctlr_port)
    logging_thd.daemon = True
    logging_thd.start()

    worker_proc = WorkerProc(req_queue, res_queue, model_name, model_weight, device_id)
    worker_proc.daemon = True
    worker_proc.start()

    print('Listening on ({}, {})'.format(ip, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        # shutdown server in another thread to avoid dead-lock
        threading.Thread(target=httpd.shutdown, daemon=True).start()
    worker_proc.terminate()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
