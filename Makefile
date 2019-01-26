PROJECT_NAME=argus-vsb-power
CONT_NAME=--name=$(PROJECT_NAME)

NET=--net=host
IPC=--ipc=host
VOLUMES=-v $(shell pwd):/workdir


all: stop build run

build:
	docker build -t $(PROJECT_NAME) .

stop:
	-docker stop $(PROJECT_NAME)
	-docker rm $(PROJECT_NAME)

logs:
	docker logs -f $(PROJECT_NAME)

run:
	nvidia-docker run --rm -it \
		$(NET) \
		$(IPC) \
		$(VOLUMES) \
		$(CONT_NAME) \
		$(PROJECT_NAME) \
		bash
