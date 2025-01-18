# Making it sparse

An autoencoder on its own is not so useful for interpreting the LLM.
Just as the LLM learned incomprehensible ways of encoding
text into 768 neural activations, the autoencoder will learn incomprehensible ways
of encoding those neural activations into its \\(F\\) features.

Here's how we encourage the autoencoder to make human-comprehensible features:

- First we make its job much easier by setting \\(F\\) to be much larger than 768.
  As we saw [earlier](training_an_autoencoder.md#training), the autoencoder can't
  perfectly preserve activations when \\(F \< 768\\), but it does ok when \\(F = 1000 \\).
  Going forwards I'll instead set \\(F = 10000 \\).
- Then we make its job harder by not letting it use very many features. We'd like to
  add a L0 penalty to the loss, i.e. we add the number of nonzero features to the loss.
  Because we need the loss to be differentiable, we instead use a L1 penalty:
  we add the sum of all the features to the loss. Since the features must be nonnegative
  because of the ReLU, this pushes most features to zero.

Q: I don't see the motivation for these steps. Why is this expected to be make the features match up
to abstract ideas like "this text contains flying"?<br>
A: It'll be less mysterious once we go through some theory on how LLMs might think.<br>
Q: Can I skip the theory and go to the experiments?<br>
A: Sure, go to the [next page](training_a_sparse_autoencoder.md).

## Theory for how LLMs might think

(See Anthropic's previous work
["Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html)
for a longer explanation.)

Assumption 1: The LLM processes information in a sparse way. That is,
it knows about thousands of human-understandable topics, but
but an individual sentence only triggers a few of them at a time.

If the input text is "Mary had a little lamb", the LLM's internal representation
is something like
`[contains_sheep = 0.5, female_character = 1, male_character = 0, contains_dog = 0, ...]`,
where the `...` is a long list of items that _aren't_ true about the story.
Let's call these hypothesized items _concepts_,
to distinguish them from the \\(F\\) definitely real _features_
in the sparse autoencoder's hidden layer. The end goal is to train the
autoencoder so that we can read off the concepts from the features.
These concepts

Assumption 2: The
[one-dimensional linear representation hypothesis](https://transformer-circuits.pub/2024/july-update/index.html#linear-representations)—the
concepts are
in fact vectors in \\(\\mathbb{R}^{768}\\), and
the LLM stores information as a linear sum
of near-perpendicular vectors.

When humans look at the activations, we just see
the sum of all concept vectors, and there's
no easy way for humans to read that representation.
No one neuron of the 768 neurons is the `contains_sheep` concept; it's spread across all neurons.

```admonish
It's possible to fit [exponentially](https://mathoverflow.net/a/24887)
[more](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) than 768 near-perpendicular vectors in 
\\(\mathbb{R}^{768}\\), so the LLM isn't limited to 768 concepts.
```

## Theory for how the sparse autoencoder should work

The sparse autoencoder has to transform the LLM activation \\(l \\in \\mathbb{R}^{768} \\) into a vector of features
where only a few features are nonzero, and then recover \\(l\\).
A plausible way to do this is:

- Use `encoder_linear` as 10000 linear probes. Each probe has two parts: \\(a_i \\in \\mathbb{R}^{768} \\)
  that points in the same direction as a concept vector, and a bias \\(b_i \\in \\mathbb{R} \\).
  Hence the ith autoencoder feature is \\( f_i = a_i \\cdot l + b_i \\). Note that \\(b_i \\) is probably negative
  (all of this is theory), so if \\( l \\) doesn't have a large component in the \\(a_i\\) direction,
  the ReLU will set the feature to 0.
- Now only a few of the features are nonzero, satisfying the L1 penalty.
- `decoder_linear` just reverses `encoder_linear`, mapping the ith feature back to the direction
  \\(a_i \\in \\mathbb{R}^{768} \\).
- This reverse mapping isn't perfect, since most features were set to 0 by the ReLU.
  But those features were associated with concepts
  that the LLM wasn't thinking about (that's the sparseness assumption). So not much of \\(l\\) is lost
  from the ReLU.

We don't know that the autoencoder will use the above algorithm.
But since this seems like the best algorithm, we hope that gradient descent will converge on it
and also do the hard work of discovering the linear probes. (I'm hand-waving a lot here.)

The autoencoder has to reuse its features across all the examples in the training data.
Hence the features must be synced with the concepts. So if we look at training examples
where a certain feature was strongly activated, we should see a specific pattern.
