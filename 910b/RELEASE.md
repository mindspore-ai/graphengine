# Release 1.0.0

## Major Features and Improvements
* Automatically dump the input and output of the abnormal operator when the network execution is abnormal;
* Realize dynamic multi-batch based on GotoLabel;
* Optimize the performance of dynamic shape；
* The dynamic resolution feature supports new scene that the network has multiple inputs and the shape of each input is different.

## Bugfixes 
* Fixed the issue that the input and output data of the AICPU operator cannot be dumped in the single-operator execution scenario.
* Fixed the execution fails in the custom AICPU operator cascading scenario.
* Fixed the issue that in the dynamic batch+dynamic AIPP scenario, the getinputformat and getinputdims parameters are inconsistent.


## Thanks to our Contributors
Thanks goes to these wonderful people: wuweikang，wangcong，weiyang，yanghaorang，xutianchun，shibeiji，zhouchao, tanghuikang, zhoulili, liujunzhu, zhengyuanhua, taoxiangdong Contributions of any kind are welcome!

Contributions of any kind are welcome!

# Release 0.7.0-beta

## Major Features and Improvements
* Conditional operator memory supports separate allocation of 4G memory space;
* In the zero-copy scenario, atomic_clean supports cleaning the memory of each part of the output when the network is multi-output;
* Support profiling of multiple levels of data in inference scenarios;
* In the online compilation scenarios, GE compilation time optimization.

## Bugfixes 
* Fix the issue that calculation result is wrong when the unknown subgraph contains conditional operations;
* Fix the issue that the hccl executor fails to load the task when the input of hccl operator is unkown shape;
* Fix the issue that allgather output is wrong when it exists in the unknown subgraph and its input is unkown shape;


## Thanks to our Contributors
Thanks goes to these wonderful people: wuweikang，wangcong，weiyang，yanghaorang，xutianchun，shibeiji，zhouchao, tanghuikang, zhoulili, liujunzhu, zhengyuanhua, taoxiangdong Contributions of any kind are welcome!

Contributions of any kind are welcome!

# Release 0.6.0-beta

## Major Features and Improvements
- GE supports function control operators such as If/Case/While/For.
- In a single operator call scenario, GE supports recording the correspondence between operators and tasks for performance commissioning.
- GE supports new operator overflow positioning solution.

## Bugfixes 
- Fix the problem that the aclmdlGetCurOutputDims interface failed to query output Dims in dynamic batch scenarios.
- Fix the problem that the operator compilation options (advanced and advanced) cannot be selected.
- Fix the problem that zero copy function cannot be performed in the scene of converging conditional operators after Data operators.
- Fix the problem that the empty graph cannot be handled.



## Thanks to our Contributors
Thanks goes to these wonderful people: 
wangcong，weiyang，yanghaorang，xutianchun，shibeiji，zhouchao, tanghuikang, zhoulili, liujunzhu, zhengyuanhua, taoxiangdong
Contributions of any kind are welcome!


# Release 0.5.0-beta

## Major Features and Improvements
- Optimize Allreduce trailing parallelism, rebuild the calculation graph dependencies, adjust the calculation order, and maximize the efficiency of calculation and gradient aggregation communication in parallel, especially in large data volume gradient aggregation and low bandwidth/large cluster scenarios You can get a bigger income.
- Advance constant folding, variable fusion, conversion operator related optimization pass to the end of the graph preparation.
- Modify memory allocation algorithm, optimize GE memory allocation, and reduce memory usage in training multi-PCS scenarios.
- Support IR composition, model compilation, inference execution in the same process.

## Bugfixes 
- Fix the bug that the graphic attribute "output_name_idx_" is not serialized to the GEIR model file, resulting in the failure of the Fast-RCNN network offline inference model generation。
- Introduce timestamp in the dump data storage directory, to ensure that the dump file generated is in a different directory each time it is executed.
- Reinforce the ParserJsonFile interface to fix the program coredump bug caused by the injection of abnormal json files.
- Fix the bug that Stream binding failure scenario and sream resource leakage.

## Thanks to our Contributors
Thanks goes to these wonderful people: 
wangcong，weiyang，yanghaorang，xutianchun，shibeiji
Contributions of any kind are welcome!

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