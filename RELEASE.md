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