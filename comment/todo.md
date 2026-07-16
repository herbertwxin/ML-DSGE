# Comments

## Questions
1. Do we want to make the network a interpolation? Or do we want more out-of-sample predictive power
2. Ebrahim Kahou et al. (2024) claim that gradient based method converge to min-norm solutions, making transverality condition unnecessary. However, our setup benefits from state-support or stability regularizer motivated by transversality.

## Fixes
1. Added over-saving penality to the loss function
2. Normalized euler residual in the loss function

## Todo
1. Add Euler residual test for NN and TI at each simulation.
2. Study parameter drift with NN
3. Testing against state-support (widen the support, remove the penalty, check Euler residual)
4. Create single simulation kit for easy demo.
5. Verify Jesus's double descent phenomenon with our setup.
