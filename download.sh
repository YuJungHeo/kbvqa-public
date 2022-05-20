#!/bin/bash

wget https://www.dropbox.com/s/59l59wj47sp93xa/kvqa.tar -P data
wget https://www.dropbox.com/s/j4l5ix5qpvfk3fw/pathquestions.tar -P data

tar -xf data/kvqa.tar -C data
tar -xf data/pathquestions.tar -C data

rm data/kvqa.tar
rm data/pathquestions.tar
