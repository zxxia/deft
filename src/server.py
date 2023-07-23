import multiprocessing as mp
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
    queue = mp.Queue()
    server_address = (ip, port)
    handler = partial(RequestHandler, queue)
    httpd = HTTPServer(server_address, handler)

    worker_proc = WorkerProc(queue)
    worker_proc.daemon = True
    worker_proc.start()

    print('Listening on ({}, {})'.format(ip, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        # shutdown server in another thread to avoid dead-lock
        threading.Thread(target=httpd.shutdown, daemon=True).start()

if __name__ == '__main__':
    main()
