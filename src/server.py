import csv
import multiprocessing as mp
import os
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer

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

class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, queue: mp.Queue, *args, **kwargs):
        self.queue = queue
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length'))
        post_data = self.rfile.read(content_length)
        self.queue.put(post_data)

def main():
    model_name = "fasterrcnn_resnet50_fpn"
    model_weight = "FasterRCNN_ResNet50_FPN_Weights"
    device_id = 1
    output_path = "."
    output_file_name = 'model_A'

    ip = 'localhost'
    port = 12345
    server_address = (ip, port)
    req_queue = mp.Queue()
    res_queue = mp.Queue()
    handler = partial(RequestHandler, req_queue)
    httpd = HTTPServer(server_address, handler)

    # set up log
    os.makedirs(output_path, exist_ok=True)
    csv_fname = os.path.join(output_path, output_file_name + ".csv")
    logging_thd = LoggingThd(csv_fname, res_queue)
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
