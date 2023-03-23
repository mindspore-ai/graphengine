#ifndef __CCE_AUTOMICADD_OPERATOR_H__
#define __CCE_AUTOMICADD_OPERATOR_H__

#include <vector>

typedef struct tagCcAutomicAddOperator {
    const char *opName;

    std::vector<uint32_t> workspaces;

    std::vector<uint32_t> outputs;

} ccAutomicAddOp_t;

static ccAutomicAddOp_t ccAutomicAddOpArray[] = {
    {"FusedBatchNorm",                               {0}, {3}   },
    {"FusedBatchNormGrad",                           {},  {1,2} },
    {"BiasAddGrad",                                  {},  {0}   },
    {"Conv2DBackpropFilter",                         {},  {0}   },
    {"L2Loss",                                       {},  {0}   },
    {"SparseSoftmaxCrossEntropyWithLogits",          {},  {0}   },
};

#endif /* __CCE_AUTOMICADD_OPERATOR_H__ */