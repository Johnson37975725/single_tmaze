## single_tmaze

This is a partial implementation of evolving hebbian neural network to solve single T-maze task described in the following paper:

[1] Soltoggio, A. [Evolutionary Advantages of Neuromodulated Plasticity in Dynamic, Reward-based Scenarios](http://andrea.soltoggio.net/data/papers/SoltoggioALife2008.pdf)

The topology search and modulatory neurons are omitted (they were unnecessary for single T-maze task); some other alterations are made in this implementation.
_Maybe I will implement modulatory neurons and double T-maze if I feel like it_.

#### To run
```bash
$ python3 ./experiment.py
```

The outputs of the above execution consist of following statistics:

|column|description|
|:--:|:--:|
|1st|number of generations|
|2nd|mean fitness of population|
|3rd|median fitness of population|
|4th|standard deviation of fitness of population|
|5th|fitness of the best individual|
|6th|fitness of the worst individual|
|7th~11th|coefficients of hebbian rule of the best individual|
|12th|mean fitness of the best individual|
|13th|mean fitness of the best individual with blocked plasticity|

Please use `--help` option to find some other options.
