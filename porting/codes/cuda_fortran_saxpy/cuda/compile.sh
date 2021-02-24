#!/bin/bash

module load pgi

pgf90 -ta=tesla:cc70 -o saxpy main.cuf 
