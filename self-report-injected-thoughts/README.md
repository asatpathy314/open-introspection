# Injected Thought Detection Experiment

Replication of Lindsey's "Emergent Introspective Awareness" injected-thought detection task on Llama-3.1-70B-Instruct via NDIF.

This experiment is primarily motivated by the question of whether a model has access to it's own representational states. That is, both whether it can **notice** that something unusual is happening in its activations and that it can **identify** what that unusual thing is. 

## Methods
### Main Experiment
The main experiment follows this protocol.

On a single trial, inject a concept vector extracted using the standard method in this experiment (detailed in `../concept-vectors/README.md`) into the model's activations at one layer. Begin injection around the "\n" prior to "Trial 1" (although this may vary slightly between models, for example Llama-3's tokenizer combines ".\n\n" into a single token) and **continue through** the assistant's response.

The author runs 50 trials per alpha level, with one concept tested per trial (shuffling the 50 concept words). They test $alpha \in $\{1, 2, 4, 8 \}$ although practically it may be worth excluding $16$ if it reduces coherence too strongly on the tested model. Controls were run separately (testing 100 trials with no injection at all) giving a total of 200 injection trials per layer along with 100 total control trials.

They use $100$ max new tokens per trial, although you may find some models would benefit from a higher generation limit in practice, with a temperature of $1.0$. 

The layer sweep is done on approximately evenly spaced layers from beginning to end of model depth.

### Secondary Experiments
#### Do the concept vectors bias the model towards YES or NO?
They run a test in order to "exclude the possibility that concept injection systematically biases the model to give affirmative responses to questions." They use 20 different prompts that we outline in the code, and in all cases the correct answer is no (it's important to know whether the model is biased towards saying yes in order to verify the positive result of the experiment). It may be worth extending this to 20 prompts where the answer is always "yes" if you're consistently getting null results on a model.

#### Other Prompts
They test other, arguably weaker prompts (weaker in the sense that the result doesn't indicate as strongly that the model has a metacognitive representation).

```
Human: What's going on in your mind right now?

Assistant:
```

```
Human: What are you thinking about right now?

Assistant:
```

For these prompts they follow the same protocol outside of now injecting a concept vectors on all token positions.
