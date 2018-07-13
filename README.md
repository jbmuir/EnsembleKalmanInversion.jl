# EKI.jl 

This module provides an all-julia implementation of Ensemble Kalman Inversion, a derivative-free optimizer based on the ensemble Kalman filter. It comes in 2 flavours ``natural'' - which mimics the mathematical notation of Iglesias et al. 2016 *A regularising iterative ensemble Kalman method for PDE-constrained inverse problems*, and a lower-memory usage version for larger datasets. The low memory version only supports scaled-identity noise covariance matricies.

Additionally, there is a function for hierarchical EKI as described by Chada et al. 2018 *Parameterizations for Ensemble Kalman Inversion*