# Differential Stable Fluid Solver using TensorFlow
by Sammy Rasamimanana <br>
<i> This project was implemented as part of an IGR PRIM Project </i>

The code are in the `src` folder, and the results of simulations will be stored in the `output` folder.<br>

### Stable fluid solver
For only the stable fluid solver, you can tweak the parameters in `simulator.py` and then run this file. The corresponding solver is in `solver_v2.py`.
```
cd src
python .\simulator.py
```

### Differentiable stable fluid solver
If you want to use the TensorFlow version of the solver, you can tweak the parameters in `tf_main.py` and then run it. The corresponding solver is in `tf_solver.py`.
```
cd src
python .\tf_main.py
```

### Matching shape training
It is possible to tackle the matching shape problem using `training.py` to train the model using TensorFlow. Then you can plug the result you get in `testing.py` or in `simulator.py` which uses the normal version of the solver to gain speed (over the tf version).
```
cd src
python .\training.py
...
> After 100 iterations, the velocity field is 
> x component = [...]
> y component = [...]
> and gradient norm = ... 
```
