import json
import multiprocessing as mp
import sys
from time import perf_counter_ns
# from ctypes import cdll

# import torch

# from tcp_utils import timestamp
# from transformer_model import is_transformer
from vision_model import is_vision_model, VisionModel


# def debug_print(msg):
#     print(msg, file=sys.stderr, flush=True)

class WorkerProc(mp.Process):
    def __init__(self, req_queue: mp.Queue, res_queue: mp.Queue,
                 model_name: str, model_weight: str, device_id: int):
        super(WorkerProc, self).__init__()
        self.req_queue = req_queue
        self.res_queue = res_queue

        # variables used in DNN inference
        if is_vision_model(model_name, model_weight):
            self.model = VisionModel(model_name, model_weight, device_id)
        else:
            raise NotImplementedError

        # self.lib = None  # so lib used to modify shared memory
        # timestamp('tcp server worker', 'init')
        #     # set up so lib
        #     if request['control'] and request['priority'] > 0:
        #         # only load library when needed
        #         self.lib = cdll.LoadLibrary(os.path.abspath("../pytcppexp/libgeek.so"))
        #     else:
        #         self.lib = None

    def run(self):
        # # set the directory for downloading models
        while True:
            request = self.req_queue.get()
            print(request)
            request = json.loads(request.decode('utf-8'))
            print('worker', request, file=sys.stderr, flush=True)

            res = self.process_request(request)
            print(res)

    def process_request(self, request):
        start_t: int = perf_counter_ns()
        # if self.lib is not None:
        #     try:
        #         suffix = os.getenv("SUFFIX", None)
        #         assert suffix is not None
        #         self.lib.setMem(1, suffix.encode())
        #         self.lib.waitForEmptyGPU()
        #     except Exception as e:
        #         debug_print(e)
        assert self.model is not None
        res = self.model.infer(request)
        print(res)
        end_t: int = perf_counter_ns()
        self.res_queue.put((start_t, end_t))
        # if self.lib is not None:
        #     try:
        #         suffix = os.getenv("SUFFIX", None)
        #         assert suffix is not None
        #         self.lib.setMem(0, suffix.encode())
        #     except Exception as e:
        #         debug_print(e)
        # max_alloc_mem_byte = torch.cuda.max_memory_allocated(self.device_id)
        # max_rsrv_mem_byte = torch.cuda.max_memory_reserved(self.device_id)
        # if self.csv_writer is not None:
        #     self.csv_writer.writerow([
        #         start_t, end_t, (end_t - start_t) / 1000000, max_alloc_mem_byte,
        #         max_rsrv_mem_byte])
        return res
