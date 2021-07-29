#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

set -e

PROJECT_HOME=${PROJECT_HOME:-$(dirname "$0")/../../}
MOUNT_PROJECT_HOME=$(cd $PROJECT_HOME || return; pwd)

DOCKER_BUILD_ENV_NAME=${MOUNT_PROJECT_HOME#*/}
DOCKER_BUILD_ENV_NAME=${DOCKER_BUILD_ENV_NAME//\//\_}
DOCKER_IMAGE_TAG=ge_build_env.1.0.9
DOCKER_IAMGE_NAME=joycode2art/turing
DOCKER_FULL_IMAGE_NAME=${DOCKER_IAMGE_NAME}:${DOCKER_IMAGE_TAG}

if [ "$(uname)" == "Darwin" ]; then
    #running on Mac OS
    docker_cmd=docker
    MOUNT_PROJECT_HOME=${MOUNT_PROJECT_HOME}
    docker_work_dir=/code/Turing/graphEngine
    docker_bash_dir=/bin/bash
elif [ "$(expr substr "$(uname -s)" 1 10)" == "MINGW32_NT" ] || [ "$(expr substr "$(uname -s)" 1 10)" == "MINGW64_NT" ]; then
    #running on Windows
    docker_cmd="winpty docker"
    MOUNT_PROJECT_HOME=/${MOUNT_PROJECT_HOME}
    docker_work_dir=//code/Turing/graphEngine  
    docker_bash_dir=//bin/bash 
elif [ "$(expr substr "$(uname -s)" 1 5)" == "Linux" ]; then
    #running on Linux
    docker_cmd=docker
    MOUNT_PROJECT_HOME=${PROJECT_HOME}
    docker_work_dir=/code/Turing/graphEngine   
    docker_bash_dir=/bin/bash
fi

function build_docker_image(){
    if test -z "$(docker images |grep ${DOCKER_IAMGE_NAME} | grep ${DOCKER_IMAGE_TAG})"; then
        $docker_cmd build -t ${DOCKER_FULL_IMAGE_NAME} ${PROJECT_HOME}/scripts/env
    else
        echo "docker image for graph engine build is build ok...."
    fi
}

function pull_docker_image(){
    $docker_cmd pull $DOCKER_FULL_IMAGE_NAME
}

function enter_docker_env(){
    if test -z "$(docker images |grep ${DOCKER_IAMGE_NAME} | grep ${DOCKER_IMAGE_TAG})"; then
        echo "please run  'ge env --pull'  to download images first!"
    elif test -z "$(docker ps -a |grep ${DOCKER_BUILD_ENV_NAME})"; then
        $docker_cmd run -p 7002:22 -p 7003:7777 --privileged=true -it -v ${MOUNT_PROJECT_HOME}:/code/Turing/graphEngine --workdir ${docker_work_dir} --name ${DOCKER_BUILD_ENV_NAME} ${DOCKER_FULL_IMAGE_NAME} ${docker_bash_dir}
    elif test -z "$(docker ps |grep ${DOCKER_BUILD_ENV_NAME})"; then
        $docker_cmd start ${DOCKER_BUILD_ENV_NAME}
        $docker_cmd exec -w ${docker_work_dir} -it ${DOCKER_BUILD_ENV_NAME} ${docker_bash_dir}
    else
        $docker_cmd exec -w ${docker_work_dir} -it ${DOCKER_BUILD_ENV_NAME} ${docker_bash_dir}
    fi
}

function resert_docker_env(){
    if test -z "$(docker ps -a |grep ${DOCKER_BUILD_ENV_NAME})"; then
        echo "no runing container for graphengine build"
    elif test -z "$(docker ps |grep ${DOCKER_BUILD_ENV_NAME})"; then
        $docker_cmd rm ${DOCKER_BUILD_ENV_NAME}
    else 
        $docker_cmd stop  ${DOCKER_BUILD_ENV_NAME}
        $docker_cmd rm ${DOCKER_BUILD_ENV_NAME}
    fi
}

function help(){
    cat <<-EOF
Usage: ge env [OPTIONS]

Prepare for docker env for build and test

Options:
    -b, --build  Build docker image
    -p, --pull   Pull  docker image
    -e, --enter  Enter container
    -r, --reset  Reset container
    -h, --help
EOF

}

function parse_args(){
    parsed_args=$(getopt -a -o bperh --long build,pull,enter,reset,help -- "$@") || {
        help
        exit 1
    }

    if [ $# -lt 1 ]; then
        pull_docker_image
        enter_docker_env
        exit 1
    fi

    eval set -- "$parsed_args"
    while true; do
        case "$1" in
            -b | --build)
                build_docker_image
                ;; 
            -p | --pull)
                pull_docker_image
                ;;
            -e | --enter)
                enter_docker_env
                ;;
            -r | --reset)
                resert_docker_env
                ;;
            -h | --help)
                help
                ;;
            --)
                shift; break;
                ;;
            *)
                help; exit 1
                ;;
        esac
        shift
    done
}

function main(){
    parse_args "$@"
}

main "$@"
set -e