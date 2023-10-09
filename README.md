# deft

## Installation
* Launch the conda virtual environment as used for the older repository
* Copy the `gpu_sched_new/gpu-sched-exp/data-set/` folder as `deft/dataset`
* Compile hook library by running  
  ```bash
  cd deft/src/hook
  make
  ```
## Sanity check
* Launch dummy server without hook library.
  ```bash
  python src/dummy_server.py
  ```
* Launch dummy server with hook library.
  ```bash
  LD_PRELOAD=src/hook/build/lib/libcuinterpose.so python src/dummy_server.py
  ```

* Build docker container (Remove `sudo` if no permission)
  ```bash
  sudo docker build . -t deft
  ```

* Run docker container (Remove `sudo` if no permission)
  ```bash
  sudo docker run -it --rm --runtime=nvidia --gpus all --name test_deft \
    -v ./dataset/rene/0000000099.png:/dataset/rene/0000000099.png \
    -e LD_PRELOAD=src/hook/build/lib/libcuinterpose.so deft
  ```

* Some debugging commands (ignore)

```
python src/controller.py --hook-ports 8080 8081 \
--model-names fasterrcnn_resnet50_fpn resnet50 --ports 12345 12346

LD_PRELOAD=./src/hook/build/lib/libcuinterpose.so python src/server.py \
--port 12345 --ctlr-ip localhost --ctlr-port 5000 --name model_A

HOOK_PORT=8081 LD_PRELOAD=./src/hook/build/lib/libcuinterpose.so \
python src/server.py --port 12346 --ctlr-ip localhost --ctlr-port 5000 \
--model-name resnet50 --model-weight ResNet50_Weights --name model_B

sudo netstat -tulpn | grep 8080
curl -i -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "run=0&sync_freq=50" http://0.0.0.0:8080
```
