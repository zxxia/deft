import json
import requests
import threading


def fcfs(req_queue, running_req):
    if req_queue:
        # do nothing about the reply
        if running_req is None:
            return False, req_queue[0]
        else:
            return False, None
    else:
        return False, None


def priority_based(req_queue, running_req):
    if req_queue:
        # sort req_queue based on priority
        req_queue = sorted(req_queue,
                           key=lambda req: req.get('priority', 0),
                           reverse=True)
        # do nothing about the reply
        if running_req is None:
            return False, req_queue[0]
        else:
            # check priority of running and target req
            if req_queue[0].get('priority', 0) > running_req.get('priority', 0):
                return True, req_queue[0]
            return False, None
    else:
        return False, None


class Scheduler:
    def __init__(self, container_ip_port, sched):
        self.container_ip_port = container_ip_port
        self.req_queue = []
        self.running_req = None
        self.lock = threading.Lock()
        if sched == "fcfs":
            self.sched = fcfs
        elif sched == 'priority_based':
            self.sched = priority_based
        else:
            raise NotImplementedError

    def recv_req(self, req)->bool:
        try:
            self._get_ip_port(req)
        except KeyError:
            return False
        with self.lock:
            self.req_queue.append(req)
        return True

    def finish_running_req(self):
        with self.lock:
            self.running_req = None

    def schedule(self):
        with self.lock:
            do_preempt, req_to_sched = self.sched(
                self.req_queue, self.running_req)
            if do_preempt:
                # stop running req
                self._preempt()
            if req_to_sched is not None:
                # schedule target req
                ip, port, hook_ip, hook_port = self._get_ip_port(req_to_sched)
                if not req_to_sched['started']:
                    _ = requests.post('http://{}:{}'.format(ip, port),
                                      json.dumps(req_to_sched).encode())

                    req_to_sched['started'] = True
                # send run signal to container
                _ = requests.post(
                    'http://{}:{}'.format(hook_ip, hook_port),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data="run=1".encode())
                self.running_req = req_to_sched
                self.req_queue.remove(req_to_sched)

    def _preempt(self):
        # preempt should be called with lock acquired
        if self.running_req is None:
            return
        _, _, hook_ip, hook_port = self._get_ip_port(self.running_req)
        _ = requests.post(
            'http://{}:{}'.format(hook_ip, hook_port),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data="run=0".encode())
        # insert the running_req back to queue
        self.req_queue.append(self.running_req)
        self.running_req = None

    def _get_ip_port(self, req):
        model_name = req.get('model_name', 'fasterrcnn_resnet50_fpn')
        ip, port, hook_ip, hook_port = self.container_ip_port[model_name]
        return ip, port, hook_ip, hook_port
