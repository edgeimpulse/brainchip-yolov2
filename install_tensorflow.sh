#!/bin/bash
set -e

ls $app > filename.txt
cat filename.txt

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
	pip3 install tensorflow==2.10.1 -f https://tf.kmtea.eu/whl/stable.html
	# pip3 install tensorflow-aarch64==2.10.1

	echo ""
	echo $PWD
	echo ""

	wget https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-arm64
	chmod +x bazelisk-linux-arm64
	ln -s `pwd`/bazelisk-linux-arm64 /usr/bin/bazel
	bazel

	cd /usr/bin
	ln -s python3 python
	apt-get install rsync

	# git clone --branch v0.18.0 https://github.com/tensorflow/addons.git

	# ls $app > filename.txt
	# cat filename.txt

	# cd /app/addons
	# python3 ./configure.py
	# bazel build build_pip_pkg
	# bazel-bin/build_pip_pkg /tmp/tensoraddons_pkg

	# pip3 install /tmp/tensoraddons_pkg/tensorflow_addons-0.18.0-cp38-cp38-linux_aarch64.whl

else
    pip3 install tensorflow==2.10.1
fi


