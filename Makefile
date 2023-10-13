all: build
.PHONY: all clean build run run_low run_high prune

NEED_ROOT := $(shell docker info > /dev/null 2> /dev/null ; echo $$?)

ifeq ($(NEED_ROOT),1)
	RUN_AS := sudo
else
	RUN_AS :=
endif

build:
	$(RUN_AS) docker build . -t deft

run:
	$(RUN_AS) docker run -it --rm --runtime=nvidia --gpus all \
	--name test_deft \
    	-v ./dataset/rene/0000000099.png:/dataset/rene/0000000099.png \
    	-e LD_PRELOAD=src/hook/build/lib/libcuinterpose.so deft

run_low:
	$(RUN_AS) docker run -it --rm --runtime=nvidia --gpus all --name deft0 \
	-v ./output:/output \
	-v ./dataset/rene/0000000099.png:/dataset/rene/0000000099.png \
	--net host \
	-e LD_PRELOAD=src/hook/build/lib/libcuinterpose.so deft \
	python3 src/server.py --output-path output/ --name model_A \
	--ctlr-ip localhost --ctlr-port 5000

run_high:
	$(RUN_AS) docker run -it --rm --runtime=nvidia --gpus all --name deft1 \
	-v ./dataset/rene/0000000099.png:/dataset/rene/0000000099.png \
	-v ./output:/output \
	--net host \
	-e HOOK_PORT=8081 \
	-e LD_PRELOAD=src/hook/build/lib/libcuinterpose.so deft \
	python3 src/server.py --output-path output/ --name model_B \
	--model-name resnet50 --model-weight ResNet50_Weights --port 12346 \
	--ctlr-ip localhost --ctlr-port 5000

clean:
	$(RUN_AS) docker kill $$($(RUN_AS) docker ps -a -q)

prune:
	$(RUN_AS) docker system prune
