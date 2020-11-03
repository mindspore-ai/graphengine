inc_path := $(shell pwd)/inc/external/
out_path := $(shell pwd)/out/graph/lib64/stub/
stub_path := $(shell pwd)/common/graph/stub/

mkdir_stub := $(shell mkdir -p $(out_path))
graph_local_stub := $(shell $(HI_PYTHON) $(stub_path)/gen_stubapi.py $(inc_path) $(out_path))
