# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
ge st test.
"""
import pytest
import subprocess
import os

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_ge
def test_resnet50_train():
    ge_st_dir=os.environ.get('GE_ST_DIR',
            '/home/jenkins/workspace/release_pkg/gate/graphengine_lib')
    ge_lib_dir=os.environ.get('GRAPHENGINE_LIB', '/home/jenkins/workspace/release_pkg/gate/graphengine_lib')

    real_pythonpath=os.environ.get('REAL_PYTHONPATH')
    pythonpath=os.environ.get('PYTHONPATH')
    if real_pythonpath:
        if pythonpath:
            os.environ['PYTHONPATH']=real_pythonpath+':'+pythonpath
        else:
            os.environ['PYTHONPATH']=real_pythonpath
    print('PYTHONPATH: '+os.environ.get('PYTHONPATH'))

    os.environ['ASCEND_OPP_PATH']='/usr/local/HiAI/runtime/ops'
    os.environ['ASCEND_ENGINE_PATH']='/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_ms_engine.so:' \
                                     '/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:' \
                                     '/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so:'+ \
                                     ge_lib_dir + '/libge_local_engine.so'
    print('ASCEND_OPP_PATH: '+os.environ.get('ASCEND_OPP_PATH'))
    print('ASCEND_ENGINE_PATH: '+os.environ.get('ASCEND_ENGINE_PATH'))
    print('LD_LIBRARY_PATH: '+os.environ.get('LD_LIBRARY_PATH'))

    cmd=ge_st_dir + '/st_resnet50_train'
    print('cmd: '+cmd)
    os.environ['SLOG_PRINT_TO_STDOUT']="1"
    ret=subprocess.call([cmd], shell=True)
    assert ret==0

