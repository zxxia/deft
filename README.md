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


* Some debugging commands (ignore)

```
sudo netstat -tulpn | grep 8080
curl -i -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "run=0&sync_freq=50" http://0.0.0.0:8080
```
