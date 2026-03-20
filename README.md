# CD-pCNL
Implementation of particle smoothing algorithms for the class of CD-SSMs; extended with local pCN(L) updates for cSMC in high dimensions


Notes for implementation:

Import generics for:
1. cSMC (forward pass and backward pass)
2. kernel building

Build generics for:
1. Euler numeric scheme
2. Transform W to X from Stanton

Workflow:
1. User defines M0 and Mt proposal kernels
2. User defines potential functions G0 and Gt 
3. If local proposals are being used - user runs a regular cSMC proposal to get the initial reference path
4. Run iterations of the cSMC algorithm

Algorithm structure
1. Propose paths in Brownian motion space; propose end points (likely via a linear Gaussian transition density)
2. Potential functions use Wt and end points to transform to X space and calculate the particle weights
3. Run regular cSMC with the above Mt and Gt structure

Proposal kernels requirements:
1. Takes extra parameters: dt = time to transpire; num = number of steps of numeric (e.g Euler) scheme; yt = observation at time t end point
2. Returns the path and endpoint

Potential modifications to cSMC algorithm imported:
1. Currently uses log PDF functions for M0 and Mt that may not be available in continuous time. Might need to rewrite this
    to explicitly function in continuous time.



TODO list:
1. Implement the Reimann sum generic
2. Implement the Delyon Hu generic
3. Write model.py to use a drift and diffusion function
4. use the drift and diffusion function to complete the Mt and Gt functions in kernel.py
5. Check with Chris what is happening at t=0 for M0 and G0
6. Refactor the csmc.py file to not use Gamma as this construct is not available in continuous time