# Steering

We want to also see a causal effect:
If an autoencoder feature is on when there's
a certain pattern in the text,
can we make that pattern appear in generated text
by turning the feature on?

Specifically, for a feature i, we set
`nudge` to be the ith vector in the autoencoder's `decoder_linear` weight matrix.
Recall that this weight matrix is a tensor with dimensions `(10000, 768)`
(number of autoencoder features, dimension of the LM's residual activations).
So `nudge` is in
\\(\\mathbb{R}^{768} \\).
Then we use the LM to generate text,
except at each  autoregressive step we
add `10 * nudge` to the residual activation
after the 2nd layer (i.e. the same layer the autoencoder was trained on).
Anecdotally I've found that
`10 * nudge` makes steering more likely to work than `nudge`,
although at the cost of some text quality. We prompt the LM with "Once upon a time".

(The above algorithm is similar to, but not the same as, the clamping
Anthropic describes
[here](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#appendix-methods-steering).)

```admonish warning
To get the ith feature vector,
It's tempting to instead define
`nudge = sae.decoder(onehot)`,
where `onehot = torch.tensor([0,...,0,1,0,...,0])` 
is a unit vector with a `1` in the ith place.

The problem is that `sae.decoder` multiplies by the weight matrix
(picking out the correct feature vector) but _also_ adds the bias.
This bias is large enough to mess up steering, i.e.
adding in this corrupted `nudge` doesn't make the model generate
text that resembles the desired feature.
```

After the initial tuning to get the scale factor 10,
I generated two steered examples (limited to 100 tokens)
for each of features 0, 1, 2, and 6, without cherrypicking the generations.
Note that I've specifically chosen feature 6 since it's an interesting case.

After generating the text, I fed it back into the unsteered LM, and used
the autoencoder to check whether the feature activated, using the same
block element format as
[here](manual_feature_examination.md#how-to-read-the-examples).
This is a sanity check: If steering pushes the generated text towards
exhibiting feature i's pattern, the text should make feature i activate.
These highlighted texts are hidden by default, but you can click to see them.

## Feature 0

I [said](manual_feature_examination.md#feature-0), "This feature activates on the quotation mark when someone starts talking". Steering is a complete failure here. If you look at the highlights, feature 0 doesn't even activate on the one quotation mark in the second generation.

> Once upon a time, there was a little boy named Tim. He went on vacation with his mom, who put on a big bed. in small jail, so she said thank you. Tim was in jail, but in her while she was. They had to get in their room and make a big, clear room. Tim was so happy to enter the room and make everything in.
> After the door,, Tim was in a room with many passages. She,thanks for her home

<details>
<summary>Click to see the version with feature activations highlighted </summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ little ▁ boy ▁ named ▁ Tim ▁. ▁ He ▁ went ▁ on ▁ vacation ▁ with ▁ his ▁ mom ▁, ▁ who ▁ put ▁ on ▁ a ▁ big ▁ bed ▁. ▁ in ▁ small ▁ jail ▁, ▁ so ▁ she ▁ said ▁ thank ▁ you ▁. ▁ Tim ▁ was ▁ in ▁ jail ▁, ▁ but ▁ in ▁ her ▁ while ▁ she ▁ was ▁. ▁ They ▁ had ▁ to ▁ get ▁ in ▁ their ▁ room ▁ and ▁ make ▁ a ▁ big ▁, ▁ clear ▁ room ▁. ▁ Tim ▁ was ▁ so ▁ happy ▁ to ▁ enter █ the ▁ room ▁ and ▁ make ▁ everything ▁ in ▁. ▁
> ▁
> ▁After ▁ the ▁ door ▁, ▁, ▁ Tim ▁ was ▁ in ▁ a ▁ room ▁ with ▁ many ▁ passages ▁. ▁ She ▁, ▁thanks ▁ for ▁ her ▁ home ▁

</details>

> Once upon a time, there was a little girl named Lily. She had a silver crown and her brother had won before.. She was a good boy, who wincedog means so smooth in."
> Lily and her brother were both so rich. They had, and the big brother made her smile. Even though she had to leave, like in jail, rich to get in with Ben.
> They both had surprise, and had to say goodbye, so they got

<details>
<summary>Click to see the version with feature activations highlighted </summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ little ▁ girl ▁ named ▁ Lily ▁. ▁ She ▁ had ▁ a ▁ silver ▁ crown ▁ and ▁ her ▁ brother ▁ had ▁ won ▁ before ▁. ▁. ▁ She ▁ was ▁ a ▁ good ▁ boy ▁, ▁ who ▁ win █ced ▁og ▁ means ▁ so ▁ smooth ▁ in ▁." ▁  ▁
> ▁
> ▁L ▁ily ▁ and ▁ her ▁ brother ▁ were ▁ both ▁ so ▁ rich ▁. ▁ They ▁ had ▁, ▁ and ▁ the ▁ big ▁ brother ▁ made ▁ her ▁ smile ▁. ▁ Even ▁ though ▁ she ▁ had ▁ to ▁ leave ▁, ▁ like ▁ in ▁ jail ▁, ▁ rich ▁ to ▁ get ▁ in ▁ with ▁ Ben ▁. ▁  ▁
> ▁
> ▁They ▁ both ▁ had ▁ surprise ▁, ▁ and ▁ had ▁ to ▁ say ▁ goodbye ▁, ▁ so ▁ they ▁ got ▁

</details>

## Feature 1

I [said](manual_feature_examination.md#feature-1),
"This feature activates on a list of concrete nouns, almost always in the context of characters wondering about this list or playing pretend with the list."
Steering is somewhat successful here: there are lists like "a cloud, a small bird" or "a game or a pretend game". And feature 1 then activates on those lists.

> Once upon a time, there was a little bird, and each day.
> One day, when it was a bit of a wind, when it saw a cloud, a small bird.
> Then, a lot of more. Then, it learned something new.
> Every day and every morning, and every night, until it had two weeks of.
> many months and months.
> Finally, when it stopped, but mom and dad.
> Then, many more.

<details>
<summary>Click to see the version with feature activations highlighted </summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ little ▁ bird ▁, ▁ and ▁ each ▁ day ▁. ▁
> ▁
> ▁One ▁ day ▁, ▁ when ▁ it ▁ was ▁ a ▁ bit ▁ of ▁ a ▁ wind ▁, ▁ when ▁ it ▁ saw ▁ a ▁ cloud ▁, ▂ a ▅ small ▆ bird ▁. ▁
> ▁
> ▁Then ▁, ▁ a ▁ lot ▁ of ▁ more ▁. ▁ Then ▁, ▁ it ▁ learned ▁ something ▁ new ▁. ▁
> ▁
> ▁Every ▁ day ▁ and ▂ every ▅ morning ▁, ▁ and ▁ every █ night ▁, ▂ until ▁ it ▁ had ▁ two ▁ weeks ▁ of ▁. ▁
> ▁ many ▁ months ▁ and ▁ months ▁. ▁
> ▁
> ▁Finally ▁, ▁ when ▁ it ▁ stopped ▁, ▁ but ▁ mom ▁ and ▁ dad ▁. ▁
> ▁
> ▁Then ▁, ▁ many ▁ more ▁. ▁

</details>

> Once upon a time, there was a wise old man who liked to travel. He liked to eat, he was very curious and he wanted a new, sweet sometimes. He loved to learn new things.
> Every time he visited a new place, he wanted to watch television, or guess some more. He sometimes played a game or a pretend game. Then he guessed a story, and made some more games.
> Eventually, it was very fun.
> "Are so many

<details>
<summary>Click to see the version with feature activations highlighted </summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ wise ▁ old ▁ man ▁ who ▁ liked ▁ to ▁ travel ▁. ▁ He ▁ liked ▁ to ▁ eat ▁, ▁ he ▁ was ▁ very ▁ curious ▁ and ▁ he ▁ wanted ▁ a ▁ new ▁, ▁ sweet ▁ sometimes ▁. ▁ He ▁ loved ▁ to ▁ learn ▁ new ▁ things ▁. ▁  ▁
> ▁
> ▁Every ▁ time ▁ he ▁ visited ▁ a ▁ new ▁ place ▁, ▁ he ▁ wanted ▁ to ▁ watch ▁ television ▁, ▁ or ▁ guess ▁ some ▁ more ▁. ▁ He ▁ sometimes ▁ played ▁ a ▁ game ▁ or ▂ a █ pretend ▂ game ▄. ▁ Then ▁ he ▁ guessed ▁ a ▁ story ▁, ▂ and ▁ made ▁ some ▁ more ▁ games ▁. ▁
> ▁
> ▁Eventually ▁, ▁ it ▁ was ▁ very ▁ fun ▁. ▁
> ▁
> ▁" ▁Are ▁ so ▁ many ▁

</details>

## Feature 2

I [said](manual_feature_examination.md#feature-2),
"This feature activates on 'time' in 'it was time to/for', usually in a phrase like 'the characters were having fun, but then it was time for something less fun'."

In the first generation, the model does generate "time to", although not after "it was". So that's a partial success. The second generation is another complete failure, and feature 2 doesn't activate anywhere on it.

> Once upon a time, there was a group to play with to a group for a long time to for the one to go to to the nation for a parade. "Time to go for the parade to for the parade to for the last to go for the Wheel. to the Wheel to the room for the Wheel. to remove the group for the tank to to exit for the to to exit.
> As for the for the group to go, the excited for to be for the new exit

<details>
<summary>Click to see the version with feature activations highlighted </summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ group ▁ to ▁ play ▁ with ▁ to ▁ a ▁ group ▁ for ▁ a ▁ long ▁ time ▁ to ▁ for ▁ the ▁ one ▁ to ▁ go ▁ to ▁ to ▁ the ▁ nation ▁ for ▁ a ▁ parade ▁. ▁ " ▁Time █ to ▁ go ▁ for ▁ the ▁ parade ▁ to ▁ for ▁ the ▁ parade ▁ to ▁ for ▁ the ▁ last ▁ to ▁ go ▁ for ▁ the ▁ Wheel ▁. ▁ to ▁ the ▁ Wheel ▁ to ▁ the ▁ room ▁ for ▁ the ▁ Wheel ▁. ▁ to ▁ remove ▁ the ▁ group ▁ for ▁ the ▁ tank ▁ to ▁ to ▁ exit ▁ for ▁ the ▁ to ▁ to ▁ exit ▁. ▁
> ▁
> ▁As ▁ for ▁ the ▁ for ▁ the ▁ group ▁ to ▁ go ▁, ▁ the ▁ excited ▁ for ▁ to ▁ be ▁ for ▁ the ▁ new ▁ exit ▁

</details>

> Once upon a time, there was a family for dinner. They all sat down for dinner for the meal to eat.
> Before dinner for dinner, for dinner to be for dinner for everyone to eat. Mom to order the family to eat to eat dinner for dinner.
> After dinner, for a full meal for dinner for the family to eat for dinner. for dinner for dessert their meal for them to be for dinner every year where they were for the dessert.
> The day to

<details>
<summary>Click to see the version with feature activations highlighted </summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ family ▁ for ▁ dinner ▁. ▁ They ▁ all ▁ sat ▁ down ▁ for ▁ dinner ▁ for ▁ the ▁ meal ▁ to ▁ eat ▁. ▁
> ▁
> ▁Before ▁ dinner ▁ for ▁ dinner ▁, ▁ for ▁ dinner ▁ to ▁ be ▁ for ▁ dinner ▁ for ▁ everyone ▁ to ▁ eat ▁. ▁ Mom ▁ to ▁ order ▁ the ▁ family ▁ to ▁ eat ▁ to ▁ eat ▁ dinner ▁ for ▁ dinner ▁. ▁
> ▁
> ▁After ▁ dinner ▁, ▁ for ▁ a ▁ full ▁ meal ▁ for ▁ dinner ▁ for ▁ the ▁ family ▁ to ▁ eat ▁ for ▁ dinner ▁. ▁ for ▁ dinner ▁ for ▁ dessert ▁ their ▁ meal ▁ for ▁ them ▁ to ▁ be ▁ for ▁ dinner ▁ every ▁ year ▁ where ▁ they ▁ were ▁ for ▁ the ▁ dessert ▁. ▁
> ▁
> ▁The ▁ day ▁ to ▁

</details>

I looked at more generations with feature 2, and "dinner" appears often.
I'm uncertain if this is coincidence or if there is an explanation
(whereas for feature 6 below, I found an explanation).

## Feature 6

In the top 10 examples (ranked by strength of activation) from the TinyStories validation set,
feature 6 only activated on "fire" (or more weakly on "firemen").
And yet these generations aren't related to fire at all:

> Once upon a time, there was an ignorant driver. One day, the cars that was traveling quickly, and so did a terrible crash. The lights went so quick
> ly, that was very loud.
> \<|endoftext|>

<details>
<summary>Click to see the version with feature activations highlighted </summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ an ▁ ignorant ▁ driver ▁. ▁ One ▁ day ▁, ▁ the ▁ cars █ that ▁ was ▁ traveling ▁ quickly ▁, ▁ and ▁ so ▁
> did ▁ a ▁ terrible ▁ crash ▇. ▁ The ▁ lights ▁ went ▅ so ▁ quickly ▁, ▁ that ▁ was ▁ very ▁ loud ▂. ▁
> ▁\<|endoftext|> ▁

</details>

> Once upon a time, there was a train. On one of the roads, a very big crash happened. It made a lot of noise. "". crash the cars and truck.," and cars," crash with a loud noise. That noise and accident.
> One day, red and noisy cars came to distant. The river, which is really a rough and dangerous. Jark.
> The people in the traffic and noise and traffic crash, ever.
> The red

<details>
<summary>Click to see the version with feature activations highlighted </summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ train ▂. ▁ On ▁ one ▁ of ▁ the ▁ roads ▁, ▁ a ▁ very ▁ big ▁ crash ▆ happened ▁. ▁ It ▁ made ▁ a ▁ lot ▁ of ▁ noise ▂. ▁ " ▁". ▁ crash ▆ the ▁ cars ▄ and ▁ truck ▁. ▁," ▁ and ▁ cars ▃," ▁ crash ▄ with ▁ a ▁ loud ▁ noise ▂. ▁ That ▁ noise ▅ and ▁ accident ▃. ▁
> ▁
> ▁One ▁ day ▁, ▁ red ▁ and ▁ noisy ▂ cars ▃ came ▁ to ▁ distant ▁. ▁ The ▁ river ▅, ▁ which ▂ is ▁ really ▁ a ▁ rough ▁ and ▁ dangerous ▂. ▁  ▁J ▂ark ▁. ▁
> ▁
> ▁The ▁ people ▁ in ▁ the ▁ traffic ▅ and ▁ noise ▄ and ▁ traffic ▆ crash █, ▁ ever ▁. ▁  ▁
> ▁
> ▁The ▁ red ▁

</details>

If you look at the highlights, you'll see that feature 6 is activating, but on
unrelated words like "crash" and "cars".
A natural hypothesis is that feature 6 isn't a "fire" feature, but instead a "dangerous thing" feature.

If we look at examples from the validation set that don't activate quite as strongly
as the top 10, we indeed see that feature 6 activates on "earthquake" and "flood":

Excerpt 1:

> " ▁Mom ▁my ▁, ▁ what ▁'s ▁ happening ▁?" ▁ she ▁ cried ▁. ▁ " ▁It ▁'s ▁ an ▁ earthquake ▆, ▁ Lily ▁," ▁ her ▁ mom ▁my ▁ said ▁

Excerpt 2:

> Tim ▁my ▁'s ▁ mom ▁my ▁ told ▁ him ▁ to ▁ come ▁ inside ▁ because ▁ there ▁ might ▁ be ▁ a ▁ flood ▇. ▁ Tim ▁my ▁ didn ▁'t ▁ know ▁ what ▁ a ▁ flood █ was ▁, ▁ but ▁ he ▁ listened ▁ to ▁ his ▁ mom ▁my ▁ and ▁ went ▁ inside ▁. ▁

I speculate that the LM thinks fire is the most dangerous of all dangerous things,
so the top-activating examples were all fire.
