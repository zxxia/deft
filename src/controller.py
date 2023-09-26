import argparse
import json
from http import HTTPStatus
from typing import Optional
from flask import Flask, request, Response

from scheduler import Scheduler

app = Flask(__name__)

scheduler: Optional[Scheduler] = None

def parse_args():
    parser = argparse.ArgumentParser(description="A DNN inference container.")
    parser.add_argument('--model-names', metavar="MODEL_NAME", nargs="+",
                        type=str, default=["fasterrcnn_resnet50_fpn"],
                        help="DNN model name to run")
    parser.add_argument('--ip', metavar='IP', type=str, default='localhost',
                        help="IP address of the container.")
    parser.add_argument('--ports', metavar='IP', nargs="+", type=int,
                        default=[12345], help="Ports containers listen to.")
    parser.add_argument('--priors', metavar='Priorities', nargs="+", type=int,
                        default=[0], help="Priorities of containers.")
    return parser.parse_args()


@app.route('/', methods=['POST'])
def distrbute_requests():
    if request.method == 'POST':
        global scheduler
        req_data = json.loads(request.get_data().decode('utf-8'))
        req_data['started'] = False

        if scheduler is not None:
            scheduler.recv_req(req_data)
            scheduler.schedule()
            return Response("Request accepted", status=HTTPStatus.ACCEPTED,
                            content_type="text")
        else:
            return Response("Scheduler is not properly set.",
                            status=HTTPStatus.INTERNAL_SERVER_ERROR,
                            content_type="text")
    else:
        raise NotImplementedError


@app.route('/job_finish', methods=['POST'])
def job_finish():
    if request.method == 'POST':
        global scheduler
        if scheduler is not None:
            scheduler.finish_running_req()
            scheduler.schedule()
            return Response("Job finish notification received",
                            status=HTTPStatus.OK,
                            content_type="text")
        else:
            return Response("Scheduler is not properly set.",
                            status=HTTPStatus.INTERNAL_SERVER_ERROR,
                            content_type="text")
    else:
        raise NotImplementedError


if __name__ == '__main__':
    args = parse_args()
    container_ip_port = {}
    for model_name, port in zip(args.model_names, args.ports):
        container_ip_port[model_name] = (args.ip, port, args.ip, args.ip + 1)
    scheduler = Scheduler(container_ip_port)
    app.run(host='0.0.0.0', port=5000)
