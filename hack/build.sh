#!/usr/bin/env bash

set -x

version=$1

PWD=$(cd "$(dirname "$0")"/../;pwd)
cd $PWD
rm -rf output
mkdir -p output
cd output
cp -r ../Makefile ./
cp -r ../backbone ./
cp -r ../hack ./
cp -r ../modelnet40 ./
cp -r ../s3dis ./
cp -r ../scannetv2 ./
cp -r ../scanobjectnn ./
cp -r ../shapenetpart ./
cp -r ../utils ./

# others
cp -r ../assets ./
cp -r ../.gitignore ./
cp -r ../LICENSE ./
cp -r ../*.md ./

cd ../

fname=PointMC${version}
mv ./output ./${fname}
zip -r ./${fname}.zip ./${fname}/*
mv ./${fname} ./output
mkdir -p release
mv ./${fname}.zip ./release
