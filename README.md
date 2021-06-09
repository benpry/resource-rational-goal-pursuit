# README

This repository contains code and data for the following paper:

Prystawski, B., Mohnert, F., Tosic, M., and Lieder, F. (2021) Resource-Rational Models of Human Goal Pursuit. *Topics in Cognitive Science*.

## Code

The "code" folder consists of five subfolders:

- "main" contains implementations of the models and microworld environments.
- "run_model" contains code that runs the model and saves the performances and exogenous inputs.
- "fitting" contains code that fits a model to the human data and aggregates the results from individual participants and models. 
- "analysis" contains Jupyter notebooks for analyzing the experimental data and simulated model data.
- "condor_jobs" contains .sub files that can be used to run the above code on an htcondor computing cluster to simulate data and fit models.

## Data

The "data" folder contains experimental data, the results of fitting the models, and simulated data from the input cost analysis, model recovery, and qualitative analysis.
