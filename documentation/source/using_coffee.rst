Using COFFEE to solve your own system of differential equations
===============================================================

Let's assume that you have a system to differential equations that you wish to
implement. It contains a time derivative, of some order, set equal to an 
equation that contains various spatial derivatives of some order.

COFFEE relies on the method of line. Thus given the value of the time
derivative of the solution to the equations a method needs to be chosen
to compute the value of the function in the future. This method is represented
by an object in the `solvers` module. If suitable code is not provided you will
need to implement your own. 

Each `solver` calls a `system` objects `evaluate()` method to get the value of the time
derivatives. Thus you will need to implement a `system` and write code
to return the value of the time derivative.

COFFEE expects every `solver` to pass an object called a `timeslice` to the
`evaluate()` method. A `timeslice` object contains the values of the function
at a specified time over a particular grid. Currently COFFEE assumes that 
`evaluate()` routines only need access to data at one point in time. If you
wish to change this you will need to ensure that the most
recent `timeslice` has a reference to previous `timeslices` or contains sufficient
data. The correct place to alter `timeslices` to achieve this is in the `solver`
and the `systems` initial data routines. Feel free to subclass the abstract base
class for `timeslice` objects, `ABCTimeSlice`, but you can also dynamically 
add class members if you want.

The `system` object will need someway to estimate spatial derivatives given
a `timeslice` that contains enough data. COFFEE provides several choices given
in the `diffop` (differential operator) module. Our example systems pick a
differential operator which the `system` is constructed, but selection can be 
performed at runtime if desired. If the existing differential operators
are not sufficient please feel free to implement your own. Please read the 
`diffop` documentation for advice on how to do this. Ultimately `system` objects
are free to estimate spatial derivatives in anyway they like with or without
`diffop` members. Feel free to hack away until you get something that works.

Once the `solver` can compute the values of the function for the next time step
it returns those values in a `timeslice`. 

All current COFFEE simulations are initiated by calling the `IBVP` classes
`run` method. This method mainly consists of a loop. Before the loop
an initial `timeslice` is constructed by calling the `system` objects
`initial_data` method. During each iteration the `system` objects `timestep`
method is called to get the next `timestep`. Each `action` is run, if the
`action` conditions are satisfied (more on this below). Then the
`solver` objects `advance()` method is called. This method does what is described
above and returns the new current time and new current `timeslice`.
At the end of each loop an `MPI.Barrier` is enforced.

And that is what you need to ensure for you own code to run! 

Assuming that existing differential operator and time integrators are sufficient
then you will only need to implement a `system` class. Take a look at the
examples and the abstract base class, `System`, for more documentation.

Lastly, please feel free to contact the maintainers. We want COFFEE to be used and
are happy to help.

