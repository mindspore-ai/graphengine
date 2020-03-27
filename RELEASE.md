# Release 0.1.0-alpha

This is the initial release of GraphEngine(GE) which was designed by the researchers and engineers in Huawei Technologies Co.,Ltd. GE is implemented via C++ and acts as a powerful backing force for MindSpore. GE is a linked up module between MindSpore front end and Ascend Chips.

## Main features

- GE API
  - GE provides an unified interface with the front end including graph management interfaces i.e., graph loading and graph execution, and GE core initiallization and finalization interfaces.

- Graph Processing
  - Six parts of graph processing operations are executed within GE, i.e. graph preparation, graph partition, graph optimization, graph compilation, graph loading and graph execution.
  - GE highly optimizes the graph defined by the front end to a form which can be effectively executed on Ascend Chips.
  - GE supports several optimizations to reduce computation costs, e.g. operator fusion, parallel operation, data format transition.

- Debugging
  - GE supports data dump and graph dump for debugging.
  - GE provides profiling tools to thoroughly analyze the speed performances.