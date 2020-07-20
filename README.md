Disclaimer: almost no empirical testing of the claims below took place so far.

### What we did

Exploration.

`lesca.py` is the result.

`python.ipynb` is its full context and kiln: an unsightly beast learns machine learning.

### What we learned

Learnable scaffolding: naturally optimize choices in arbitrary programs. A convenient way to do everything. Python library.

A function call can call a sequence of other functions. In all programs, we have trees and sequences. We could use a recursive/recurrent neural net to deliver end-to-end differentiable estimates to arbitrary choices inside arbitrary programs. We do this here.

We don't have the compute power (nor a good-enough implementation, nor good-enough datasets) to do non-toy experiments, though.

It may sound like a worthless random thing from this description.

### AGI

A quick synonym game tells you that AGI is arbitrary-program optimization. A search for everything.    
Prior general approaches (Bayesian optimization, evolution, local search…) can do that, but fail to deliver good solutions quickly enough.    
Recently, practical optimization approaches have emerged under the name "machine learning" (ML). They are mostly domain-specific, but correctly routing information in arbitrary programs (as our scaffolding approach suggests) should make ML perfectly general *and* practical. The effect is rather small now, though in the past, ML-based approaches *have* unexpectedly and completely dominated fields.

Optimization requires goal functions — **1** learnable number to be optimized (exactly as our `choice` is), predicting the global information with local info, possibly many times in one execution. (Note that our framework supports both single-goal and many-goals optimization.)

We don't rely on a particular semantics, but instead route information through global variables, which allows the principles of our framework to transfer to any programming environment.

Both generating and changing programs can be viewed as some discrete choices, arranged hierarchically in some tree, DAG, or whatever a semantics requires. (Our scaffolding framework permits any hierarchies: in a function, hierarchies and sequences of calls are shadowed by an RNN.)

In formal analyses of AGI, it's common to analyse the average case. But arbitrary-program optimization picks the best instead of a random choice, so such an analysis is useless. It can do anything and be anything, as long as its basic functionality set is Turing-complete and efficient.

(As an aside, the very preliminary results hint at very deep connections between our approach and the human mind (with high-level slow often-single-threaded thoughts which still use billions of neurons). But, anecdotal. Requires significantly more exporation.)

It seems to us that there are no fundamental breakthroughs in understanding required on this path to AGI, only refinements. A lot of refinements: currently, we do not have *any* good benchmark results, so a naive analysis would naturally dismiss this effort.

Hypothesis:

- Advanced representations (such as gradual dependent types, homoiconicity, self-knowledge, and interoperability) are not required for general intelligence (and, by themselves, will not lead to it). We can get away with merely augmenting generic computation with sufficiently rich routing of differentiable information. If other things are useful, they'll be learned, and function engineering only harms progress in AGI.

Future work includes:

- Efficiency: a GPU-accelerated implementation; partial evaluation.

- Evaluating the approach on searching for [Metamath](http://us.metamath.org/index.html) proofs, and on [AutoML](https://openml.github.io/automlbenchmark/).

- Opening all functions to generation, including a very simple partially-evaluated Turing-complete functionality base, having but subverting all concreteness. It's the only way to learn *everything* end-to-end. This allows to choose choices, choose self, use and train self-awareness, create cooperative swarms of algorithms, play tic-tac-toe, and do everything else that generality allows. Intelligent choices open up a lot of possibilities that are unfeasible with nearly-random chaos.

(Admittedly, some of these are fundamental breakthroughs in and of themselves. But not insurmountable.)

Of course, all help would be appreciated.

---

Now, if you'll excuse me, I need to go commune with the Elder Gods.
