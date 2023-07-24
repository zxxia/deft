import csv
import multiprocessing as mp
import os
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer

from worker_proc import WorkerProc


class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, queue: mp.Queue, *args, **kwargs):
        self.queue = queue
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length'))
        post_data = self.rfile.read(content_length)
        self.queue.put(post_data)

def main():
    ip = 'localhost'
    port = 12345
    req_queue = mp.Queue()
    res_queue = mp.Queue()
    server_address = (ip, port)
    handler = partial(RequestHandler, req_queue)
    httpd = HTTPServer(server_address, handler)
    model_name = "fasterrcnn_resnet50_fpn"
    model_weight = "FasterRCNN_ResNet50_FPN_Weights"
    device_id = 1
    output_path = "."
    output_file_name = 'model_A'
    # set up log
    os.makedirs(output_path, exist_ok=True)
    csv_fname = os.path.join(output_path, output_file_name + ".csv")
    csv_fh = open(csv_fname, 'w', 1)
    csv_writer = csv.writer(csv_fh, lineterminator='\n')
    csv_writer.writerow(
        ['start_timestamp_ns', 'end_timestamp_ns', 'jct_ms',
         'max_allocated_gpu_memory_allocated_byte',
         'max_reserved_gpu_memory_byte'])

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
