import argparse
import json
import requests
from flask import Flask, request, Response

app = Flask(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="A DNN inference container.")
    parser.add_argument('--model-names', metavar="MODEL_NAME", nargs="+",
                        type=str, default=["fasterrcnn_resnet50_fpn"],
                        help="DNN model name to run")
    parser.add_argument('--ip', metavar='IP', type=str, default='localhost',
                        help="IP address of the container.")
    parser.add_argument('--ports', metavar='IP', nargs="+", type=int,
                        default=[12345], help="Ports containers listen to.")
    return parser.parse_args()


@app.route('/', methods=['POST'])
def distrbute_requests():
    if request.method == 'POST':
        req_data = json.loads(request.get_data().decode('utf-8'))
        model_name = req_data.get('model_name', 'fasterrcnn_resnet50_fpn')
        ip, port = app.config['container_ip_port'].get(
            model_name, ('localhost', 12345))
        print(model_name, ip, port)
        reply = requests.post('http://{}:{}'.format(ip, port),
                              request.get_data())
        return Response(reply.text, status=reply.status_code,
                        content_type=reply.headers['content-type'])


if __name__ == '__main__':
    args = parse_args()
    container_ip_port = {}
    for model_name, port in zip(args.model_names, args.ports):
        container_ip_port[model_name] = (args.ip, port)
    app.config['container_ip_port'] = container_ip_port
    app.run(host='0.0.0.0', port=5000)
