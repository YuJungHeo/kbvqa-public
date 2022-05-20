#!/bin/bash

wget https://www.dropbox.com/s/bw1w9hoe8qxqj61/ht_pq_checkpoints.tar -P ckpt
tar -xf ckpt/ht_pq_checkpoints.tar -C ckpt
rm ckpt/ht_pq_checkpoints.tar
