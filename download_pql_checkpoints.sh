#!/bin/bash

wget https://www.dropbox.com/s/q45fmazwn3r8hqw/ht_pql_checkpoints.tar -P ckpt
tar -xf ckpt/ht_pql_checkpoints.tar -C ckpt
rm ckpt/ht_pql_checkpoints.tar
