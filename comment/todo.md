# Comments

## Questions
1. Do we want to make the network a interpolation? Or do we want more out-of-sample predictive power -  The Jesus' double descent problem
2. ~~Ebrahim Kahou et al. (2024) claim that gradient based method converge to min-norm solutions, making transverality condition unnecessary. However, our setup benefits from state-support or stability regularizer motivated by transversality.~~
3. ~~Does neural network learn better with one residual in the loss or it does not matter.~~ The network learns much better with one residual in the loss.
4. How do we handle multiple optimality conditions in the loss function.

## Fixes
1. Added over-saving penality to the loss function
2. Normalized euler residual in the loss function

## Todo
- [ ] Add Euler residual test for NN and TI at each simulation.
- [ ] Study parameter drift with NN
- [ ] Testing against state-support (widen the support, remove the penalty, check Euler residual)
- [x] Create single simulation kit for easy demo.
- [ ] Check for steady-state formulation in deterministic model for RE
- [ ] Verify Jesus's double descent phenomenon with our setup.
