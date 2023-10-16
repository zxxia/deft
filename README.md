# deft

## Installation
* Launch the conda virtual environment as used for the older repository
* Copy the `gpu_sched_new/gpu-sched-exp/data-set/` folder as `deft/dataset`

## Test
* Open 4 terminals

* Build docker container
  ```bash
  cd deft
  make
  ```

* In terminal 0, run a container listening for low-priority requests
  ```bash
  cd deft
  make run_low
  ```
* In terminal 1, run a container listening for high-priority requests
  ```bash
  cd deft
  make run_high
  ```

* In terminal 2, run deft controller which listens to all inference requests
  and schedule GPU usage for containers
  ```bash
  cd deft
  python src/controller.py --hook-ports 8080  8081 \
    --model-names fasterrcnn_resnet50_fpn resnet50 --ports 12345 12346
  ```

* In terminal 3, send inference requests
  ```bash
  cd deft
  ./scripts/send_req.sh
  ```

* Find inference logs in `output/`

## Sanity check without containers
* Compile hook library by running  
  ```bash
  cd deft/src/hook
  make
  ```
* Launch dummy server without hook library.
  ```bash
  cd deft
  python src/dummy_server.py
  ```
* Launch dummy server with hook library.
  ```bash
  LD_PRELOAD=src/hook/build/lib/libcuinterpose.so python src/dummy_server.py
  ```

* Plot
  ```bash
  cd deft
  python src/plot_jct_timeseries.py \
    --model-A-jct-log output/model_A.csv \
    --model-B-jct-log output/model_B.csv
  ```

* Some debugging commands (ignore)

```
sudo netstat -tulpn | grep 8080
curl -i -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "run=0&sync_freq=50" http://0.0.0.0:8080
```
