# Automatic feature examination

Manually judging if features are interpretable
doesn't scale. So I've outsourced the work
to GPT-4o, with the following prompts:

System prompt:

> You are a helpful assistant.

User prompt:

> I'll send you a sequence of texts,
> where tokens have been highlighted using Unicode Block Elements.
> Look for a pattern in which tokens get strong highlights.
> Rank how clear the pattern is on a 1-5 scale, where 1 is no detectable pattern,
> 3 is a vague pattern with some exceptions, and 5 is a clear pattern with no exceptions.
> Focus on the pattern in the strong highlights. Describe the pattern in 10 words or less"

Let's also ask GPT-4o to rank how
interpretable the original language model activations are
(same prompt).
Hence we can compare to the sparse autoencoder's features,
which ideally GPT-4o will rate as much more clear.

```admonish
Finicky detail: The LM activations can be positive or negative, 
unlike the always nonnegative autoencoder features. 
We want the activations to be nonnegative, so that
0 will be "weakly activating" and a large positive number
will be "strongly activating".

Three ways we could do this:
- Take the ReLU, i.e. set negative activations to 0
- Take the absolute value
- Use an affine shift, i.e. add the smallest possible number
  to all the acitvations so that the least activation is now 0
  
I don't think there's a strong justification for choosing one
of these methods over another, so I've implemented them all.
```

Results:

![GPT-4o's ranking of sparse autoencoder features and LM neurons](assets/llm_and_sae_clearness.svg)

## Discussion

The results are directionally as expected: GPT-4o ranks the
autoencoder features as clearer: less likely to be 3/5 than the LLM activations, more likely to be 5/5.

It would be nice to have a reliable automated interpretability system,
which would allow extensions like "
if we trained for half as long, is the autoencoder half as interpretable?"
But my inclination is to not fully trust my current automated system because of the caveats
discussed below.

### Caveat 1

What I call "language model activations"
are the residual stream activations,
i.e. the exact tensors I trained the SAE on
Anthropic instead
compared their autoencoder's interpretability
to neurons in Claude's previous MLP layer.

There's an argument that
residual stream is
less likely to have interpretable activations as-is,
since the addition mixes the outputs of
earlier neurons together.
("As-is" means that no autoencoder/mechanistic interpretability tools are required.)

My prior is that the MLP neurons for `roneneldan/TinyStories-33M` wouldn't be interpretable either,
but that's definitely worth looking into.

### Caveat 2

My anecdotal impression from looking through the results is that GPT-4o is
a lenient grader. For instance, GPT-4o gave a clearness
score of 4/5 to the LM's 0th residual activation (the affine case), and it
found the pattern "Strong highlights emphasize dialogue and reactions."
I disagree with this score, since a typical example of this LM activation
has most tokens strongly highlighted, unlike the autoencoder's sparsity. For example:

> Once ▄ upon ▃ a ▃ time ▄, ▄ there ▄ was ▄ a ▄ huge ▃ bear ▃ called ▄ Peter ▄. ▄ Peter ▄ lived ▃ deep ▂ in ▄ the ▄ forest ▅ in ▄ a ▄ big ▃, ▄ cos ▄y ▂ cave ▄. ▄ ▄ ▄One ▄ day ▅, ▄ Peter ▄ was ▄ feeling ▃ tired ▅ and ▄ decided ▄ he ▄ should ▃ go ▅ to ▆ sleep ▆. ▄ So ▅ he ▅ climbed ▄ into ▆ his ▅ bed ▆ and ▅ closed ▄ his ▄ eyes ▆. ▄ Suddenly ▅, ▅ he ▅ heard ▆ a ▆ noise ▆ outside █ his ▄ cave ▅. ▄ " ▅What ▅ was ▆ that ▄?" ▄ he ▅ thought ▄. ▅ ▅ ▅Peter ▄ went ▄ outside ▆ to ▅ investigate ▅. ▅ He ▆ saw ▅ a ▄ little ▄ mouse ▃, ▄ and ▄ the ▅ mouse ▄ was ▄ in ▃ trouble ▅! ▄ Its ▄ tail ▄ was ▄ stuck ▃ under ▄ a ▃ big ▄ rock ▄ and ▄ it ▄ couldn ▄'t ▅ get ▄ free ▄. ▄ ▅ ▄" ▅Don ▄'t ▃ worry ▃," ▃ said ▅ Peter ▄. ▄ " ▅I ▄'ll ▂ help ▅ you ▃!" ▄ He ▅ used ▅ all ▄ his ▅ strength ▃ to ▄ move ▅ the ▄ huge ▄ rock ▄ and ▅ the ▅ mouse ▃ ran ▄ away ▄. ▄ ▅ ▄Peter ▄ smiled ▃, ▄ but ▄ he ▅ was ▄ very ▄ tired ▄ now ▄. ▄ He ▅ said ▅ goodbye ▅ to ▆ the ▅ mouse ▂ and ▄ went ▄ back ▅ to ▅ his ▅ cave ▅. ▄ He ▅ was ▄ so ▄ exhausted ▄, ▃ he ▄ fell ▃ asleep ▅ right ▄ away ▄. ▄ ▅ ▄From ▄ that ▃ day ▄ on ▄, ▄ Peter ▄ and ▅ the ▅ mouse ▃ were ▄ good ▃ friends ▃. ▄ They ▅ always ▄ looked ▄ out ▇ for ▄ each ▃ other ▃ and ▄ helped ▄ each ▂ other ▄ if ▁ they ▅ got ▄ into ▅ trouble ▄. ▄ And ▄ they ▅ both ▄ slept ▄ well ▄ each ▃ night ▅, ▄ knowing ▄ they ▄ had ▄ a ▄ friend ▄ who ▅ would ▃ help ▄ them ▄ out ▄ if ▂ they ▅ needed ▅ it ▄! ▄

The strongest activating token in the above example is "outside" in "he heard a noise outside", which is neither a dialogue or a reaction.

What could have gone wrong here?

- Maybe more prompt engineering is needed. I could tell GPT-4o "be strict" or
  "if most tokens have strong highlights, rate it 1/5".
- Maybe reading Unicode block elements is trickier for an LLM, and a different format
  (e.g. keep the activation strength as a floating point number)
  might be easier for it to comprehend.
- Or maybe the prompt should be split in two:
  - First ask GPT-4o to describe a pattern given the examples
  - Then ask it (with a blank context) to decide which of two examples (a real one and a distractor) exhibits the pattern. Ideally this would detect when GPT-4o has wrongly found a generic pattern that could match any example
