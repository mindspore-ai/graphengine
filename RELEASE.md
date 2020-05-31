# Release 0.3.0-alpha

## Major Features and Improvements
- It supports dynamic batches and shapes with certain fixed levels.([!22](https://gitee.com/mindspore/graphengine/pulls/22))
- Scope fusion interfaces are opened allowing user defined scope fusion rules.([!24](https://gitee.com/mindspore/graphengine/pulls/22))
- Enhance the maintenance and measurement capability.([!28](https://gitee.com/mindspore/graphengine/pulls/24))
- A package of compiled libraries is generated after compilation to facilitate code deployment.([!21](https://gitee.com/mindspore/graphengine/pulls/21))

## Bugfixes 
- Fix the bug that the interface of GE IR construction operator does not support dynamic input in the middle of the ordinary input port.([!24](https://gitee.com/mindspore/graphengine/pulls/24))
- Fix checkpoint subgraph validation and data callback process to resolve the problem that checkpoint could not be generated in some scenarios.([!28](https://gitee.com/mindspore/graphengine/pulls/28))
- When MindSpore is compiled in a directory involving symbolic links, GE log records real path of which code was built when executing testcases using installed whl package.([!16](https://gitee.com/mindspore/graphengine/pulls/16), [!489](https://gitee.com/mindspore/mindspore/pulls/489))
- Find third-party software in specified directories only.([!18](https://gitee.com/mindspore/graphengine/pulls/18))




## Thanks to our Contributors
Thanks goes to these wonderful people: 
wangcong，weiyang，yanghaorang，xutianchun，shibeiji
Contributions of any kind are welcome!

# Release 0.2.0-alpha

## Major Features and Improvements
- Provides a common graph-level option, and multiple requirements can also share this mechanism in the future.
- Improve graph compilation performance.
- Optimize memory allocation.
- Optimize serveral operators e.g., Slice, StridedSlice, ScatterMax etc.

## Bugfixes 
- Delete redudant codes.([#I1EU2Z](https://gitee.com/mindspore/graphengine/issues/I1EU2Z))
- Fix HCCL initilization bugs under train and eval scenarios.([#I1DIBJ](https://gitee.com/mindspore/graphengine/issues/I1DIBJ))
- Optimize compilation and linking process, enhancing efficiency and performance of concurrent compilation of GraphEngine and MindSpore. ([#I1DFIY](https://gitee.com/mindspore/mindspore/issues/I1DFIY))
- Fix the bug that GE checkpoint cannot save variable names correctly.([#I1DIBJ](https://gitee.com/mindspore/graphengine/issues/I1DIBJ))
- Save dump files on every iteration instead of every execution.([#I1DIBJ](https://gitee.com/mindspore/graphengine/issues/I1DIBJ))


## Thanks to our Contributors
Thanks goes to these wonderful people: Wang Cong, Tianchun Xu, Haoran Yang.

Contributions of any kind are welcome!
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