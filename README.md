# Differential Stable Fluid Solver using TensorFlow
by Sammy Rasamimanana <br>
<i> This project was implemented as part of an IGR PRIM Project </i>

The code are in the `src` folder, and the results of simulations will be stored in the `output` folder.<br>

## TO DO
<li> Advanced UI using PyQt5 (ok)
<li> Merge UI and fluid simulation
<li> Merge UI and animation layers
<li> Change the speed of the trajectory

## Progress
### Week 29/03
<li> Code optimization of the TensorFlow pipeline
<li> Rewrite boundary conditions on centered grid

### Week 05/04
<li> Adding intermediate velocity constraints
<li> Tests on some constraints

### Week 12/04
<li> Created basic UI to simplify tests
<li> Parsing file to .json format to make tests more organized
<li> Changing initial condition to be guided by the trajectory

### Week 12/04
<li> Change loss function using cosine loss
<li> Change initial guess
<li> Try training dt
<li> Add load function in UI

### Week 26/04
<li> Try on larger grid
<li> Try curl approach
<li> Try vortex approach

### Week 15/05
<li> Tests with manually placed vortices
<li> Tests with adding vortices at specific frames 

### Week 22/05
<li> Rewrite using Staggered Grid
<li> Add BC variable (dirichlet or neumann)
<li> Forcing using LU decomposition to speed up (No Sparse Tensor solver yet in tf...)
<li> Training using Staggered Grid and neumann
<li> Project the initial velocity field 
<li> Rewrite loss function so that it uses only tensors
