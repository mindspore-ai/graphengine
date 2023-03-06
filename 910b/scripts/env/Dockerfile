# this dockerfile used for graphengine build
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

FROM ubuntu:18.04
RUN apt-get update \
	&& apt-get install -y git g++ wget unzip clang-format-9 build-essential lcov vim

# install for doxygen
RUN apt-get install -y graphviz doxygen 

# install for graph ensy engine
RUN cpan install -y Graph::Easy

RUN wget https://cmake.org/files/v3.16/cmake-3.16.7-Linux-x86_64.tar.gz 

RUN mkdir -p /opt/cmake-3.16.7 \
	&& tar -xvf cmake-3.16.7-Linux-x86_64.tar.gz -C /opt/cmake-3.16.7 --strip-components=1 \
	&& ln -sf  /opt/cmake-3.16.7/bin/*  /usr/bin/ \
  	&& mv /usr/bin/clang-format-9 /usr/bin/clang-format

RUN wget https://github.com/ccup/lcov/archive/refs/tags/add_lcov.tar.gz -O add_lcov.tar.gz \
	&& mkdir -p /opt/addlcov1.0.0 \
	&& tar -xvf add_lcov.tar.gz -C /opt/addlcov1.0.0 \
	&& mv /opt/addlcov1.0.0/lcov-add_lcov/bin/lcov /usr/bin/addlcov 

ENV PROJECT_HOME=/code/Turing/graphEngine

RUN mkdir /var/run/sshd
RUN echo "root:root" | chpasswd
RUN sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22 7777

RUN useradd -ms /bin/bash debugger
RUN echo "debugger:ge123" | chpasswd

CMD ["/usr/sbin/sshd" "-D" "&"]

RUN echo "alias ge=/code/Turing/graphEngine/scripts/ge.sh">>~/.bashrc

