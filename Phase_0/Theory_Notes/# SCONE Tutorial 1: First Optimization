# SCONE Tutorial 1: First Optimization

## Objective

The goal of Tutorial 1 is to become familiar with the SCONE optimization workflow by optimizing a simple jumping motion.

Instead of manually designing the controller, SCONE automatically searches for controller parameters that maximize the jump height.

---

## Optimization Pipeline

Each optimization consists of four major components:

### Model

The **Model** represents the biomechanical system that is simulated.

Example:

```text
ModelOpenSim3 {
    model_file = models/H0918v3.osim
}
```

The model defines:

- Body segments
- Joints
- Muscles
- Physical properties

The optimizer never changes the model itself.

---

### Controller

The **Controller** determines how the muscles are activated.

Example:

```text
FeedForwardController
```

The controller contains parameters that the optimizer is allowed to modify.

Example:

```text
control_point_y
control_point_dt
```

These parameters define the activation pattern applied to the muscles during the jump.

---

### Objective (Measure)

The **Objective** defines what the optimizer tries to maximize or minimize.

Example:

```text
JumpMeasure
```

The objective rewards solutions that produce a higher jump.

Typical parameters include:

- prepare_time
- termination_height
- terminate_on_peak

---

### Optimizer

The **Optimizer** searches for better controller parameters.

Example:

```text
CmaOptimizer
```

SCONE uses the CMA-ES algorithm, which repeatedly:

1. Generates candidate parameter sets.
2. Simulates the movement.
3. Evaluates the fitness.
4. Learns from the best solutions.
5. Generates better candidates.

This process continues until convergence.

---

## Parameter Format

Optimization parameters are written as

```text
value ~ std <min,max>
```

Example:

```text
control_point_y = 0.3~0.01<0,1>
```

where

- **0.3** → initial mean value
- **0.01** → initial standard deviation
- **0** → minimum allowed value
- **1** → maximum allowed value

Only these parameters are optimized.

---

## Mean and Standard Deviation

During optimization:

- **Mean** is the current best estimate of a parameter.
- **Standard deviation (std)** controls how much exploration is performed.

Large std

- more exploration

Small std

- fine tuning around the current solution.

---

## Result

After many simulations, CMA-ES finds controller parameters that produce a better jump than the initial guess.

The user does not manually tune parameters; instead, the optimizer automatically improves them.

---

## Key Idea

Tutorial 1 demonstrates the complete SCONE optimization loop:

Model

↓

Controller

↓

Simulation

↓

Objective Evaluation

↓

CMA-ES Optimization

↓

Improved Controller

# SCONE Tutorial 2A: Piecewise Constant Controller

## Objective

Tutorial 2A investigates how the controller representation influences optimization performance.

Instead of using only two control points, the controller uses a larger Piecewise Constant function.

---

## Piecewise Constant Controller

The controller output is represented as several constant segments.

Example:

```text
PieceWiseConstant {
    control_points = 3
}
```

Each control point specifies:

- activation level
- duration

The optimizer searches for the best values of every control point.

---

## Control Parameters

Each control point contains two optimized parameters.

```text
control_point_y
```

Muscle activation level.

```text
control_point_dt
```

Time interval until the next control point.

Both are optimized automatically by CMA-ES.

---

## Parameter Initialization

The optimizer starts from

```text
value ~ std <min,max>
```

Example

```text
0.3~0.01<0,1>
```

meaning

- initial value = 0.3
- std = 0.01
- range = [0,1]

---

## Increasing Control Points

Compared to Tutorial 1:

- more control points
- more optimization variables
- greater controller flexibility

Advantages

- smoother control
- potentially better jumps

Disadvantages

- larger optimization problem
- longer optimization time

---

## Result

The optimizer learns a more flexible activation pattern, allowing better jumping performance than a simple controller.


# SCONE Tutorial 2B: Polynomial Controller

## Objective

Instead of Piecewise Constant control, Tutorial 2B represents muscle activation using a polynomial.

---

## Polynomial Controller

Example

```text
Polynomial {
    degree = 2
}
```

The controller output is

$$
f(t)=at^2+bt+c
$$

where

- coefficient2 = a
- coefficient1 = b
- coefficient0 = c

---

## Optimization Variables

The optimizer adjusts

```text
coefficient0
coefficient1
coefficient2
```

instead of multiple control points.

---

## Advantages

Compared with Piecewise Constant:

- fewer parameters
- smoother activation curve
- smaller optimization problem

---

## Limitations

A low-degree polynomial cannot represent complex activation patterns.

Higher flexibility requires either

- higher polynomial degree
- Piecewise Constant control.

---

## Result

Tutorial 2B demonstrates how changing the controller representation changes the optimization behavior.


# SCONE Tutorial 2C: Composite Objective

## Objective

Tutorial 2C introduces multiple optimization objectives simultaneously.

Instead of maximizing only jump height, the optimizer must also maintain a desired body posture.

---

## Composite Measure

Example

```text
CompositeMeasure
```

Several objective functions are combined into one fitness value.

---

## Jump Measure

```text
JumpMeasure
```

Rewards:

- higher jump
- successful completion

---

## DOF Measure

```text
DofMeasure {
    dof = pelvis_tilt
}
```

Penalizes excessive backward leaning.

The optimizer therefore searches for solutions that are both

- high jumping
- physically reasonable

---

## Initial Parameters

Tutorial 2C initializes optimization using

```text
init {
    file = ...
    std_factor = 5
}
```

This loads a previously optimized solution and starts searching around it.

The parameter

```text
std_factor
```

controls how much exploration is performed around the loaded solution.

---

## Multi-objective Optimization

The total fitness becomes

```
Fitness

=

Jump Score

+

Posture Score
```

A controller producing a high jump but poor posture receives a lower overall score.

---

## Result

Tutorial 2C demonstrates that optimization can satisfy multiple objectives simultaneously rather than maximizing only a single performance measure.

This produces more realistic and stable movements.