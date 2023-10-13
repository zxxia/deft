import json
import multiprocessing as mp
from time import perf_counter_ns
from vision_model import is_vision_model, VisionModel


class WorkerProc(mp.Process):
    def __init__(self, req_queue: mp.Queue, res_queue: mp.Queue,
                 model_name: str, model_weight: str, device_id: int):
        super(WorkerProc, self).__init__()
        self.req_queue = req_queue
        self.res_queue = res_queue
        self.model_name = model_name
        self.model_weight = model_weight
        self.device_id = device_id

    def run(self):
        # variables used in DNN inference
        if is_vision_model(self.model_name, self.model_weight):
            self.model = VisionModel(self.model_name, self.model_weight, self.device_id)
        else:
            raise NotImplementedError
        while True:
            request = self.req_queue.get()
            request = json.loads(request.decode('utf-8'))

            res = self.process_request(request)
            print('done')

    def process_request(self, request):
        start_t: int = perf_counter_ns()
        assert self.model is not None
        res = self.model.infer(request)
        end_t: int = perf_counter_ns()
        req_recv_t = request['req_recv_timestamp_ns']
        self.res_queue.put((start_t, end_t, req_recv_t))
        # max_alloc_mem_byte = torch.cuda.max_memory_allocated(self.device_id)
        # max_rsrv_mem_byte = torch.cuda.max_memory_reserved(self.device_id)
        return res
