#!/bin/bash

# Generate data for the abstraction example
python data_gen.py --N 10 --corr 1.0 --output $SCRATCH/output # data_75816d80b9
python data_gen.py --N 100 --corr 1.0 --output $SCRATCH/output # data_330fb6f722
python data_gen.py --N 1000 --corr 1.0 --output $SCRATCH/output # data_45dd5039c4
python data_gen.py --N 10000 --corr 1.0 --output $SCRATCH/output # data_2a6573283c

python data_gen.py --N 10 --corr 0.5 --output $SCRATCH/output # data_ee008efcbe
python data_gen.py --N 100 --corr 0.5 --output $SCRATCH/output # data_ff9207638e
python data_gen.py --N 1000 --corr 0.5 --output $SCRATCH/output # data_15ced18281
python data_gen.py --N 10000 --corr 0.5 --output $SCRATCH/output # data_1c0bd2630b

python data_gen.py --N 10 --corr 0.0 --output $SCRATCH/output # data_22e2ad4db8
python data_gen.py --N 100 --corr 0.0 --output $SCRATCH/output # data_874127f863
python data_gen.py --N 1000 --corr 0.0 --output $SCRATCH/output # data_074a994dd0
python data_gen.py --N 10000 --corr 0.0 --output $SCRATCH/output # data_f8159c8081