## Introduction

This repository has the code for the paper Explaining Rankings with Hidden Group Bonuses. See the following for some information on how to run it, and about the codebase if you want to use and modify it.

## Synthetic data generation
Synthetic data can be generated using the files in the folder synthetic.
The file `generate_additive.py` constructs instances of g-GBLR, and takes in command line arguments to determine the size of the instance, in the order n, d, k, g. The number of points given an additive bonus is equal to k * n, so k should be a floating point value between 0.0 to 1.0.

As an example, `python generate_additive.py 10000 2 0.5 2` generates an instance of 10000 tuples with 2 dimensional features, with 5000 tuples given an additive bonus, where there are 2 groups of distinct additive bonus.

The file `generate_arbitrary.py` constructs instances of SGBLR, and takes in command line arguments to determine the size of the instance, in the order n, d, k.

As an example, 

```bash
python generate_arbitrary.py 5000 5 0.1
``` 

generates an instance of 5000 tuples with 5 dimensional features, with 500 tuples given an additive bonus.

The output is written to a file in the same directory, with the parameters contained in the filename.

The files `generate_additive.py` and `generate_arbitrary.py` generate data from a uniform distribution over the weights and tuple attributes. The files `generate_zipf_additive.py` and `generate_zipf_arbitrary.py` can be used to generate data where the weights and from a Zipfian distribution instead; the usage is identical to that described above.

The seed is fixed to allow reproducibility.

## JEE Data

The JEE dataset is found in the folder JEE.

The raw data is found in `jee2009.csv`, and the preprocessed data file, which has removed the unused attributes and candidates with missing attributes in `jee.in`. 

To generate instances of different `n`, run `jeecreate.py` with the desired value of `n` as command line argument. For example, for 50000 tuples, one can run the command 

```bash
python jeecreate.py 50000
```

Note that to create instances with only one group, edit the `jeecreate.py` file and modify the code to set `CATEGORY = False` instead of `True`.


## Solver code

All solver code takes in input from standard input, and writes to standard output, and does not take any command line parameters. See Usage/Running code for a simple example of how to create a synthetic dataset and run it.

The file `ermb_2d.py` is to solve SGBLR using the ERMB algorithm. Note that it functions only for 2-dimensional input as it exploits specific properties to allow a 1-dimensional search, which is faster.

The file `ermb_kd.py` is to solve SGBLR using the ERMB algorithm for arbitrary dimension inputs.

The file `localsearch_group.py` is to solve g-GBLR with the localsearch algorithm.

The file `localsearch_singleton.py` is to solve SGBLR with the localsearch algorithm.

The file `ilp_base.py` is to solve g-GBLR with the ILPbase formulation.

The file `ilp_base_singleton.py` is to solve SGBLR with the ILPbase formulation.

The file `ilp_refined.py` is to solve g-GBLR with the ILPrefined formulation.

The file `ilp_refined_singleton.py` is to solve SGBLR with the ILPrefined formulation.

The file `ordinal_regression.py` is to solve any instance of the problem with ordinal regression. 

The file `logistic_regression.py` is to solve any instance of the problem with logistic regression.

Python files with the name format `*_functions.py` are intended to be compiled with mypyc for speedup; however the code will still work without this.

## Usage

### Setup
First, pip install according to the `requirements.txt` file. Most important is `gurobipy` (version 13 or more preferred) and `numpy`.
Note that you will require a Gurobi license (academic or otherwise) set up to be able to run the experiments. See [their website](https://www.gurobi.com/academics) for information on free licenses.

Optionally, files such as `ermb_functions.py` should be compiled once by running `mypyc ermb_functions.py` to get a closer running time to our experiments.

### Running code

For all of the Python files to run a solving method, the input should be given from stdin.

An example of how to create a synthetic dataset of 1-GBLR with $n = 10000, d = 2, k = 1000$ and run it. All solver codes function similarly, so to run ILPbase, simply replace `ilp_refined.py` with `ilp_base.py`.

```bash
python synthetic/generate_additive.py 10000 2 0.1 1
python code/ilp_refined.py < 1group_10000_2_0.1.txt
```

```bash
cd JEE
python jee_create.py 50000
cd ..
python code/ilp_refined.py < JEE/jee_50000.in
```

Outputs are written to standard output.

## Creating your own input instances
If using with other input, the general input format expected is as follows
The first line should be a line containing `n`, followed by `g` integers representing the size of each group.
The next `n` lines consist of `d` values separated by a space. The ith line represents the attributes of the ith point.

Finally, `n` more lines follow. Each line should consist of two integers, first the point, then a space, then its rank.
