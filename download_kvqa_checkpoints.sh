#!/bin/bash

wget https://www.dropbox.com/s/d2refu3w04rn5vj/ht_kvqa_checkpoints.tar -P ckpt
tar -xf ckpt/ht_kvqa_checkpoints.tar -C ckpt
rm ckpt/ht_kvqa_checkpoints.tar
