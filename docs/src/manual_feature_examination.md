# Manual feature examination

Now we have a trained autoencoder with a low
reconstruction loss, and each
feature is sparse—it only activates on a tiny
fraction of input tokens.

But what we want is each input feature to be
human-understandable and monosemantic.
That is, each feature of the autoencoder should
activate on input text with _one_ narrow theme, like
"cake/pie" or "numerical digits".

To demonstrate that this actually happens, I'll show the ten
examples from the `roneneldan/TinyStories` validation set
where the feature activates most strongly.
I'll use features 0, 1, and 2 (out of the 10000 available)
to avoid cherrypicking.

A more thorough investigation
(e.g.
[this section](https://transformer-circuits.pub/2023/monosemantic-features/index.html#feature-arabic)
of "Towards Monosemanticity")
would also check that
the long tail of weaker activations
shares the same theme.

## How to read the examples

```admonish
Each example is long, often > 200 words.
Hence for readability I've hidden the examples 
and only display the excerpt 
where the feature activated most strongly. 
You can click on an example to expand it, 
```

Suppose we've chosen a feature index (in this subsection 13)
and a text from the validation set:

> The sun was shining brightly and the birds were singing happily

We pass the text through the TinyStories language model (LM) and
pass the LM's residual state activation through
the autoencoder. Thus we obtain a list telling us how strongly
feature 13 activated on each token:

```python
[
      0.0,
      0.0,
      0.0,
      0.0,
      78.38675689697266,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      173.66055297851562,
]
```

To make this easier to read at a glance, I've used
[block elements](https://en.wikipedia.org/wiki/block_elements)
to represent feature strength:

> The ▁ sun ▁ was ▁ shining ▁ brightly ▄ and ▁ the ▁ birds ▁ were ▁ singing ▁ happily █

Here feature 13 activated once just after the LM saw
"brightly", and once after "happily".

# Sparse Autoencoder Features

### Feature 0

My interpretation: This feature activates on the quotation mark when someone starts talking.
I'm not sure if the feature is aware of the context,
or if it's specific to the character like `“`.

GPT-4o's interpretation: "Strong on exclamations and excitement"

(I sometimes disagree with GPT-4o's interpretations—see
the [next page](automatic_feature_examination.md) for a discussion.)

```admonish
I think the weird artifacts like `�` are caused by the tokenizer struggling
with quotation marks.
```

Example 1: `he ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁Wow ▁ thank ▁ you ▁ so ▁ much ▁!`

<details>
<summary>Click to see all of example 1</summary>

> Ben ▁ was ▁ a ▁ happy ▁ three ▁ year ▁ old ▁ boy ▁ who ▁ loved ▁ to ▁ explore ▁ his ▁ garden ▁. ▁ One ▁ day ▁, ▁ he ▁ noticed ▁ a ▁ little ▁ bee ▁ buzzing ▁ around ▁ the ▁ flowers ▁. ▁ He ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Hello ▁, ▁ little ▁ bee ▁. ▁â ▁€ ▁\\n ▁\\n ▁The ▁ bee ▁ flew ▁ up ▁ to ▁ him ▁ and ▁ buzz ▁ed ▁ in ▁ a ▁ friendly ▁ way ▁. ▁ Ben ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁What ▁ are ▁ you ▁ doing ▁ here ▁? ▁â ▁€ ▁  ▁\\n ▁\\n ▁The ▁ bee ▁ flew ▁ away ▁ and ▁ returned ▁ a ▁ moment ▁ later ▁, ▁ carrying ▁ a ▁ drop ▁ of ▁ sweet ▁ honey ▁ in ▁ its ▁ tiny ▁ mouth ▁. ▁ Ben ▁â ▁€ ▁™ ▁s ▁ eyes ▁ lit ▁ up ▁ and ▁ he ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁Wow ▁ thank ▁ you ▁ so ▁ much ▁! ▁ I ▁ love ▁ honey ▁! ▁â ▁€ ▁\\n ▁\\n ▁The ▁ bee ▁ flew ▁ away ▁ again ▁ and ▁ returned ▁ with ▁ some ▁ more ▁ honey ▁. ▁ Ben ▁ was ▁ filled ▁ with ▁ joy ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Oh ▁, ▁ thank ▁ you ▁ so ▁ much ▁ lovely ▁ bee ▁! ▁ I ▁ love ▁ you ▁ too ▁! ▁â ▁€ ▁\\n ▁\\n ▁Just ▁ then ▁, ▁ a ▁ big ▁ dark ▁ cloud ▁ passed ▁ overhead ▁ and ▁ Ben ▁ felt ▁ a ▁ little ▁ bit ▁ miserable ▁. ▁ The ▁ bee ▁ flew ▁ up ▁ to ▁ him ▁ and ▁ buzz ▁ed ▁ in ▁ a ▁ comforting ▁ way ▁. ▁ Ben ▁ understood ▁ what ▁ the ▁ bee ▁ was ▁ trying ▁ to ▁ tell ▁ him ▁ and ▁ said ▁ � ▁� ▇€ ▁� ▁� ▁Thank ▁ you ▁, ▁ little ▁ bee ▁. ▁ I ▁ love ▁ you ▁. ▁ That ▁ helps ▁ me ▁ feel ▁ better ▁. ▁â ▁€ ▁\\n ▁\\n ▁The ▁ cloud ▁ soon ▁ passed ▁ and ▁ Ben ▁ and ▁ the ▁ bee ▁ went ▁ back ▁ to ▁ playing ▁ and ▁ collecting ▁ honey ▁. ▁ Ben ▁ knew ▁ he ▁ would ▁ always ▁ have ▁ the ▁ little ▁ bee ▁ with ▁ him ▁ in ▁ times ▁ of ▁ sadness ▁ and ▁ that ▁ made ▁ him ▁ feel ▁ very ▁ loved ▁ and ▁ happy ▁. ▁

</details>

Example 2: `He ▁ smiled ▁, ▁ saying ▁, ▁ � ▁� █€ ▁� ▁� ▁Well ▁, ▁ I ▁â ▁€ ▁™ ▁m ▁ glad ▁ it ▁ wasn ▁â ▁€ ▁™ ▁t`

<details>
<summary>Click to see all of example 2</summary>

> Dog ▁ and ▁ Cat ▁ were ▁ best ▁ friends ▁ who ▁ did ▁ everything ▁ together ▁. ▁ Every ▁ day ▁ they ▁ would ▁ get ▁ up ▁ and ▁ play ▁ in ▁ the ▁ garden ▁. ▁\\n ▁\\n ▁One ▁ day ▁, ▁ Dog ▁ looked ▁ over ▁ and ▁ saw ▁ something ▁ disgusting ▁ on ▁ the ▁ ground ▁. ▁ He ▁ pointed ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁What ▁â ▁€ ▁™ ▁s ▁ that ▁? ▁â ▁€ ▁ Cat ▁ looked ▁ over ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁It ▁â ▁€ ▁™ ▁s ▁ gross ▁! ▁ Let ▁â ▁€ ▁™ ▁s ▁ not ▁ go ▁ near ▁ it ▁. ▁â ▁€ ▁\\n ▁\\n ▁The ▁ next ▁ day ▁ when ▁ Dog ▁ and ▁ Cat ▁ woke ▁ up ▁, ▁ they ▁ decided ▁ to ▁ play ▁ in ▁ the ▁ garden ▁ again ▁. ▁ But ▁ when ▁ they ▁ got ▁ there ▁, ▁ Dog ▁ saw ▁ the ▁ same ▁ disgusting ▁ thing ▁. ▁ He ▁ asked ▁ Cat ▁ again ▁, ▁ � ▁� ▇€ ▁� ▁� ▁What ▁â ▁€ ▁™ ▁s ▁ that ▁? ▁â ▁€ ▁ But ▁ this ▁ time ▁ Cat ▁ said ▁ nothing ▁. ▁\\n ▁\\n ▁Suddenly ▁, ▁ Dog ▁ felt ▁ angry ▁. ▁ He ▁ pointed ▁ at ▁ the ▁ thing ▁ and ▁ shouted ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Why ▁ did ▁ you ▁ bring ▁ this ▁ here ▁!? ▁â ▁€ ▁ Cat ▁ looked ▁ embarrassed ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁I ▁ wanted ▁ to ▁ show ▁ you ▁ something ▁ I ▁ found ▁ yesterday ▁, ▁ but ▁ now ▁ it ▁â ▁€ ▁™ ▁s ▁ all ▁ gross ▁. ▁â ▁€ ▁\\n ▁\\n ▁Dog ▁ was ▁ surprised ▁ and ▁ relieved ▁. ▁ He ▁ smiled ▁, ▁ saying ▁, ▁ � ▁� █€ ▁� ▁� ▁Well ▁, ▁ I ▁â ▁€ ▁™ ▁m ▁ glad ▁ it ▁ wasn ▁â ▁€ ▁™ ▁t ▁ something ▁ bad ▁. ▁â ▁€ ▁ Dog ▁ and ▁ Cat ▁ hugged ▁ and ▁ went ▁ back ▁ to ▁ playing ▁. ▁

</details>

Example 3: `Her ▁ mom ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁Why ▁ don ▁â ▁€ ▁™ ▁t ▁ we ▁`

<details>
<summary>Click to see all of example 3</summary>

> M ▁agg ▁ie ▁ and ▁ her ▁ mom ▁ were ▁ out ▁ at ▁ the ▁ store ▁ shopping ▁. ▁ Maggie ▁â ▁€ ▁™ ▁s ▁ mom ▁ was ▁ looking ▁ at ▁ some ▁ fancy ▁ dresses ▁. ▁ She ▁ pointed ▁ to ▁ one ▁, ▁ saying ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Look ▁ at ▁ this ▁ one ▁, ▁ Maggie ▁. ▁ Isn ▁â ▁€ ▁™ ▁t ▁ it ▁ nice ▁? ▁â ▁€ ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁Yes ▁, ▁ Mom ▁my ▁, ▁ it ▁â ▁€ ▁™ ▁s ▁ pretty ▁, ▁â ▁€ ▁ Maggie ▁ said ▁. ▁ � ▁� ▇€ ▁� ▁� ▁It ▁â ▁€ ▁™ ▁s ▁ got ▁ a ▁ zip ▁, ▁ too ▁. ▁ I ▁ like ▁ it ▁. ▁â ▁€ ▁\\n ▁\\n ▁Her ▁ mom ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁Why ▁ don ▁â ▁€ ▁™ ▁t ▁ we ▁ get ▁ it ▁ for ▁ you ▁? ▁â ▁€ ▁\\n ▁\\n ▁M ▁agg ▁ie ▁ was ▁ very ▁ excited ▁ and ▁ jumped ▁ up ▁ and ▁ down ▁. ▁ But ▁ then ▁ her ▁ mom ▁ noticed ▁ a ▁ sign ▁ that ▁ said ▁ � ▁� ▇€ ▁� ▁� ▁No ▁ Load ▁s ▁â ▁€ ▁. ▁ Maggie ▁ saw ▁ the ▁ sign ▁ too ▁ and ▁ asked ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Mom ▁my ▁, ▁ why ▁ can ▁â ▁€ ▁™ ▁t ▁ we ▁ load ▁ the ▁ dress ▁? ▁â ▁€ ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁It ▁ means ▁, ▁ Maggie ▁, ▁â ▁€ ▁ her ▁ mom ▁ said ▁ gently ▁, ▁ � ▁� ▇€ ▁� ▁� ▁that ▁ we ▁ can ▁â ▁€ ▁™ ▁t ▁ take ▁ it ▁ home ▁ with ▁ us ▁. ▁ We ▁â ▁€ ▁™ ▁ll ▁ have ▁ to ▁ leave ▁ it ▁ here ▁ today ▁. ▁ Maybe ▁ next ▁ time ▁ we ▁ can ▁ get ▁ it ▁. ▁â ▁€ ▁\\n ▁\\n ▁M ▁agg ▁ie ▁ was ▁ sad ▁ but ▁ she ▁ promised ▁ she ▁ wouldn ▁â ▁€ ▁™ ▁t ▁ forget ▁ the ▁ fancy ▁ dress ▁ and ▁ the ▁ zip ▁. ▁ She ▁ knew ▁ she ▁ would ▁ have ▁ to ▁ come ▁ back ▁ to ▁ the ▁ store ▁ soon ▁ so ▁ she ▁ could ▁ get ▁ the ▁ dress ▁ she ▁ wanted ▁. ▁

</details>

Example 4: `The ▁ girl ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁Okay ▁! ▁`

<details>
<summary>Click to see all of example 4</summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ a ▁ charming ▁ little ▁ girl ▁ went ▁ for ▁ a ▁ walk ▁ in ▁ the ▁ forest ▁. ▁ Suddenly ▁, ▁ she ▁ saw ▁ a ▁ snake ▁. ▁ She ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Oh ▁ no ▁! ▁â ▁€ ▁ and ▁ started ▁ to ▁ frown ▁. ▁\\n ▁\\n ▁But ▁ then ▁ the ▁ snake ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Don ▁â ▁€ ▁™ ▁t ▁ be ▁ scared ▁. ▁ I ▁â ▁€ ▁™ ▁m ▁ a ▁ friendly ▁ snake ▁! ▁â ▁€ ▁ The ▁ girl ▁ was ▁ surprised ▁, ▁ but ▁ she ▁ kept ▁ on ▁ frown ▁ing ▁. ▁ The ▁ snake ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁You ▁ can ▁ trust ▁ me ▁. ▁ I ▁ promise ▁ not ▁ to ▁ hurt ▁ you ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁The ▁ girl ▁ hesitated ▁, ▁ then ▁ the ▁ snake ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁If ▁ you ▁ want ▁, ▁ I ▁ can ▁ show ▁ you ▁ around ▁ the ▁ forest ▁. ▁ It ▁â ▁€ ▁™ ▁s ▁ very ▁ beautiful ▁. ▁â ▁€ ▁ The ▁ girl ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁Okay ▁! ▁â ▁€ ▁\\n ▁\\n ▁The ▁ girl ▁ and ▁ the ▁ snake ▁ went ▁ for ▁ a ▁ walk ▁ and ▁ had ▁ a ▁ wonderful ▁ time ▁. ▁ The ▁ snake ▁ showed ▁ her ▁ all ▁ the ▁ charming ▁ things ▁ in ▁ the ▁ forest ▁ and ▁ the ▁ girl ▁ was ▁ no ▁ longer ▁ afraid ▁. ▁\\n ▁\\n ▁At ▁ the ▁ end ▁ of ▁ the ▁ walk ▁, ▁ the ▁ girl ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Thank ▁ you ▁ for ▁ showing ▁ me ▁ around ▁ and ▁ making ▁ me ▁ feel ▁ less ▁ scared ▁. ▁â ▁€ ▁ The ▁ snake ▁ replied ▁, ▁ � ▁� ▇€ ▁� ▁� ▁It ▁ was ▁ my ▁ pleasure ▁. ▁ Come ▁ back ▁ and ▁ visit ▁ me ▁ anytime ▁. ▁â ▁€ ▁ The ▁ girl ▁ smiled ▁ and ▁ went ▁ happily ▁ on ▁ her ▁ way ▁. ▁

</details>

Example 5: `and ▁ then ▁ he ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁I ▁ think ▁ it ▁â ▁€ ▁™ ▁s ▁ one ▁`

<details>
<summary>Click to see all of example 5</summary>

> Once ▁ there ▁ was ▁ a ▁ little ▁ boy ▁ named ▁ Joe ▁. ▁ He ▁ was ▁ only ▁ three ▁ years ▁ old ▁ and ▁ loved ▁ to ▁ tell ▁ jokes ▁. ▁ One ▁ day ▁, ▁ Joe ▁ and ▁ his ▁ mom ▁ were ▁ walking ▁ home ▁ from ▁ the ▁ park ▁ when ▁ they ▁ saw ▁ a ▁ huge ▁ hot ▁ air ▁ balloon ▁. ▁  ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁Look ▁, ▁ Mom ▁my ▁, ▁â ▁€ ▁ Joe ▁ said ▁, ▁ pointing ▁ at ▁ the ▁ balloon ▁. ▁ � ▁� ▇€ ▁� ▁� ▁It ▁â ▁€ ▁™ ▁s ▁ so ▁ big ▁. ▁â ▁€ ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁Yes ▁, ▁ it ▁ certainly ▁ is ▁, ▁â ▁€ ▁ said ▁ his ▁ mom ▁. ▁ � ▁� ▇€ ▁� ▁� ▁Let ▁â ▁€ ▁™ ▁s ▁ guess ▁ how ▁ high ▁ it ▁ is ▁. ▁â ▁€ ▁\\n ▁\\n ▁Joe ▁ thought ▁ for ▁ a ▁ moment ▁ and ▁ then ▁ he ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁I ▁ think ▁ it ▁â ▁€ ▁™ ▁s ▁ one ▁ thousand ▁ feet ▁. ▁â ▁€ ▁\\n ▁\\n ▁His ▁ mom ▁ laughed ▁. ▁ � ▁� ▇€ ▁� ▁� ▁That ▁â ▁€ ▁™ ▁s ▁ too ▁ high ▁, ▁ Joe ▁. ▁â ▁€ ▁\\n ▁\\n ▁Joe ▁ was ▁ disappointed ▁, ▁ but ▁ he ▁ wanted ▁ to ▁ prove ▁ he ▁ was ▁ right ▁. ▁ He ▁ remembered ▁ the ▁ joke ▁ he ▁ heard ▁ at ▁ the ▁ park ▁ earlier ▁. ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁I ▁ know ▁ a ▁ joke ▁ that ▁ will ▁ tell ▁ us ▁, ▁â ▁€ ▁ he ▁ said ▁ excited ▁ly ▁. ▁ He ▁ rec ▁ited ▁ the ▁ joke ▁ he ▁ remembered ▁, ▁ making ▁ his ▁ mom ▁ laugh ▁. ▁\\n ▁\\n ▁Suddenly ▁, ▁ the ▁ balloon ▁ began ▁ to ▁ drift ▁ away ▁. ▁ It ▁ was ▁ getting ▁ higher ▁ and ▁ higher ▁. ▁  ▁\\n ▁\\n ▁Joe ▁ and ▁ his ▁ mom ▁ ran ▁ to ▁ catch ▁ up ▁ to ▁ it ▁, ▁ but ▁ the ▁ balloon ▁ moved ▁ too ▁ fast ▁. ▁ Joe ▁ looked ▁ at ▁ his ▁ mom ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁I ▁ guess ▁ I ▁ was ▁ right ▁ - ▁ it ▁ is ▁ one ▁ thousand ▁ feet ▁ high ▁! ▁â ▁€ ▁\\n ▁\\n ▁Joe ▁â ▁€ ▁™ ▁s ▁ mom ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁You ▁ were ▁ right ▁, ▁ Joe ▁. ▁ That ▁â ▁€ ▁™ ▁s ▁ one ▁ hot ▁ joke ▁

</details>

Example 6: `Mary ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁We ▁ came ▁ to ▁ see ▁ the ▁ bird ▁`

<details>
<summary>Click to see all of example 6</summary>

> Mary ▁ and ▁ her ▁ big ▁ brother ▁ were ▁ playing ▁ in ▁ the ▁ park ▁ one ▁ sunny ▁ day ▁. ▁ As ▁ they ▁ ran ▁ around ▁, ▁ they ▁ spotted ▁ something ▁ shiny ▁ in ▁ the ▁ sky ▁. ▁ Mary ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁What ▁ is ▁ that ▁? ▁â ▁€ ▁ Her ▁ brother ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁It ▁â ▁€ ▁™ ▁s ▁ a ▁ bird ▁, ▁ but ▁ it ▁ isn ▁â ▁€ ▁™ ▁t ▁ flying ▁, ▁ it ▁ is ▁ diving ▁! ▁â ▁€ ▁  ▁\\n ▁\\n ▁Mary ▁ and ▁ her ▁ brother ▁ decided ▁ to ▁ follow ▁ the ▁ bird ▁ and ▁ see ▁ where ▁ it ▁ was ▁ going ▁. ▁ After ▁ a ▁ short ▁ walk ▁, ▁ they ▁ arrived ▁ at ▁ an ▁ old ▁, ▁ wooden ▁ house ▁ with ▁ a ▁ big ▁ attic ▁. ▁ The ▁ bird ▁ had ▁ d ▁ived ▁ into ▁ the ▁ attic ▁ and ▁ was ▁ now ▁ perched ▁ on ▁ a ▁ window ▁ sill ▁, ▁ pre ▁ening ▁ its ▁ feathers ▁. ▁  ▁\\n ▁\\n ▁When ▁ Mary ▁ and ▁ her ▁ brother ▁ knocked ▁ on ▁ the ▁ door ▁, ▁ an ▁ old ▁, ▁ deaf ▁ woman ▁ answered ▁. ▁ She ▁ wasn ▁â ▁€ ▁™ ▁t ▁ surprised ▁ to ▁ see ▁ the ▁ bird ▁, ▁ but ▁ was ▁ surprised ▁ to ▁ see ▁ the ▁ children ▁. ▁ She ▁ asked ▁, ▁ � ▁� ▇€ ▁� ▁� ▁What ▁ are ▁ you ▁ doing ▁ here ▁? ▁â ▁€ ▁ Mary ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁We ▁ came ▁ to ▁ see ▁ the ▁ bird ▁ in ▁ your ▁ attic ▁! ▁â ▁€ ▁  ▁\\n ▁\\n ▁The ▁ old ▁ woman ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Come ▁ in ▁. ▁ The ▁ bird ▁ comes ▁ to ▁ visit ▁ me ▁ every ▁ day ▁. ▁ I ▁ can ▁â ▁€ ▁™ ▁t ▁ hear ▁ it ▁ singing ▁, ▁ but ▁ I ▁ love ▁ watching ▁ it ▁ dive ▁ down ▁ from ▁ the ▁ sky ▁. ▁â ▁€ ▁ Mary ▁ and ▁ her ▁ brother ▁ thanked ▁ the ▁ old ▁ woman ▁ and ▁ waved ▁ goodbye ▁ as ▁ they ▁ left ▁. ▁  ▁\\n ▁\\n ▁Mary ▁ and ▁ her ▁ brother ▁ had ▁ a ▁ lovely ▁ day ▁, ▁ thanks ▁ to ▁ the ▁ bird ▁ in ▁ the ▁ old ▁ woman ▁â ▁€ ▁™ ▁s ▁ attic ▁. ▁

</details>

Example 7: `Mom ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁I ▁ know ▁`

<details>
<summary>Click to see all of example 7</summary>

> Andy ▁ was ▁ playing ▁ in ▁ the ▁ park ▁. ▁ He ▁ saw ▁ a ▁ p ▁uddle ▁, ▁ and ▁ he ▁ was ▁ really ▁ excited ▁. ▁ He ▁ ran ▁ over ▁ to ▁ the ▁ p ▁uddle ▁ and ▁ said ▁ to ▁ Mom ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Look ▁ at ▁ this ▁! ▁â ▁€ ▁ But ▁ Mom ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁No ▁, ▁ Andy ▁, ▁ you ▁ can ▁â ▁€ ▁™ ▁t ▁ play ▁ here ▁. ▁ You ▁â ▁€ ▁™ ▁ll ▁ get ▁ all ▁ wet ▁. ▁â ▁€ ▁\\n ▁\\n ▁Andy ▁ was ▁ sad ▁, ▁ and ▁ he ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁But ▁ I ▁ want ▁ to ▁ play ▁ in ▁ it ▁! ▁ Ple ▁ee ▁e ▁ase ▁! ▁â ▁€ ▁ Mom ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁I ▁â ▁€ ▁™ ▁m ▁ sorry ▁, ▁ but ▁ you ▁ can ▁â ▁€ ▁™ ▁t ▁. ▁ It ▁â ▁€ ▁™ ▁s ▁ too ▁ dangerous ▁. ▁â ▁€ ▁\\n ▁\\n ▁Andy ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁A ▁ww ▁ww ▁. ▁ That ▁â ▁€ ▁™ ▁s ▁ not ▁ fair ▁. ▁â ▁€ ▁ Mom ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁I ▁ know ▁. ▁ I ▁â ▁€ ▁™ ▁m ▁ sorry ▁. ▁ You ▁ can ▁ play ▁ in ▁ the ▁ sandbox ▁ instead ▁. ▁ That ▁ will ▁ be ▁ safer ▁. ▁â ▁€ ▁\\n ▁\\n ▁Andy ▁ wasn ▁â ▁€ ▁™ ▁t ▁ sure ▁ if ▁ he ▁ wanted ▁ to ▁, ▁ but ▁ then ▁ he ▁ remembered ▁ how ▁ much ▁ fun ▁ it ▁ was ▁ to ▁ play ▁ in ▁ sand ▁. ▁ So ▁ he ▁ decided ▁ to ▁ go ▁ and ▁ play ▁ in ▁ the ▁ sandbox ▁ instead ▁. ▁ He ▁ was ▁ still ▁ a ▁ little ▁ bit ▁ sad ▁, ▁ but ▁ the ▁ thought ▁ of ▁ playing ▁ in ▁ the ▁ sand ▁ soon ▁ made ▁ him ▁ excited ▁ again ▁. ▁  ▁ He ▁ ran ▁ off ▁ to ▁ the ▁ sandbox ▁, ▁ ready ▁ to ▁ have ▁ some ▁ fun ▁. ▁

</details>

Example 8: `T ▁oby ▁ nodded ▁ in ▁ understanding ▁ and ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁Be ▁ brave ▁`

<details>
<summary>Click to see all of example 8</summary>

> T ▁oby ▁ was ▁ a ▁ young ▁ boy ▁, ▁ he ▁ was ▁ very ▁ curious ▁. ▁ One ▁ day ▁ as ▁ he ▁ was ▁ playing ▁ in ▁ the ▁ park ▁ he ▁ noticed ▁ a ▁ cricket ▁ ch ▁ir ▁ping ▁ loudly ▁. ▁ Toby ▁ wondered ▁ what ▁ the ▁ cricket ▁ was ▁ saying ▁. ▁ He ▁ also ▁ noticed ▁ that ▁ the ▁ cricket ▁ looked ▁ quite ▁ nervous ▁. ▁ Toby ▁ had ▁ many ▁ questions ▁ so ▁ he ▁ decided ▁ to ▁ ask ▁. ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁Hello ▁ little ▁ cricket ▁," ▁ Toby ▁ said ▁. ▁\\n ▁\\n ▁The ▁ cricket ▁ stopped ▁ ch ▁ir ▁ping ▁, ▁ looked ▁ around ▁ and ▁ said ▁ � ▁� ▇€ ▁� ▁� ▁yes ▁, ▁ what ▁ is ▁ it ▁? ▁â ▁€ ▁  ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁Why ▁ do ▁ you ▁ look ▁ so ▁ nervous ▁? ▁â ▁€ ▁ Toby ▁ asked ▁. ▁\\n ▁\\n ▁The ▁ cricket ▁ took ▁ a ▁ deep ▁ breath ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁I ▁ need ▁ to ▁ go ▁ somewhere ▁ but ▁ I ▁'m ▁ scared ▁ to ▁ leave ▁." ▁\\n ▁\\n ▁T ▁oby ▁ nodded ▁ in ▁ understanding ▁ and ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁Be ▁ brave ▁ little ▁ cricket ▁! ▁ I ▁'m ▁ sure ▁ you ▁'ll ▁ make ▁ it ▁ there ▁ safe ▁. ▁â ▁€ ▁\\n ▁\\n ▁The ▁ cricket ▁ smiled ▁ weak ▁ly ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Thank ▁ you ▁! ▁ Now ▁ I ▁ just ▁ need ▁ to ▁ find ▁ the ▁ way ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁T ▁oby ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Maybe ▁ I ▁ can ▁ help ▁ you ▁. ▁ Would ▁ you ▁ like ▁ that ▁? ▁â ▁€ ▁  ▁\\n ▁\\n ▁The ▁ cricket ▁ nodded ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Yes ▁, ▁ I ▁ would ▁ appreciate ▁ it ▁ very ▁ much ▁! ▁â ▁€ ▁\\n ▁\\n ▁So ▁ Toby ▁ and ▁ the ▁ cricket ▁ began ▁ walking ▁ together ▁. ▁ Toby ▁ was ▁ filled ▁ with ▁ wonder ▁ and ▁ admiration ▁ for ▁ the ▁ brave ▁ little ▁ cricket ▁. ▁ With ▁ Toby ▁ as ▁ his ▁ guide ▁, ▁ the ▁ cricket ▁ was ▁ no ▁ longer ▁ nervous ▁ - ▁ he ▁ was ▁ feeling ▁ brave ▁ and ▁ confident ▁. ▁ Toby ▁ was ▁ happy ▁ he ▁ could ▁ help ▁ the ▁ cricket ▁ on ▁ his ▁ journey ▁. ▁

</details>

Example 9: `He ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁No ▁, ▁ let ▁â ▁€ ▁™ ▁s ▁ go ▁`

<details>
<summary>Click to see all of example 9</summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ were ▁ two ▁ children ▁. ▁ One ▁ child ▁ was ▁ clever ▁, ▁ the ▁ other ▁ was ▁ very ▁ foolish ▁. ▁ They ▁ were ▁ walking ▁ along ▁ the ▁ path ▁ to ▁ school ▁ when ▁ they ▁ passed ▁ by ▁ a ▁ garden ▁. ▁ The ▁ clever ▁ child ▁ said ▁ to ▁ the ▁ foolish ▁ one ▁, ▁ � ▁� ▇€ ▁� ▁� ▁We ▁ should ▁ poke ▁ that ▁ garden ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁But ▁ the ▁ foolish ▁ one ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁No ▁, ▁ don ▁â ▁€ ▁™ ▁t ▁ be ▁ silly ▁. ▁ That ▁â ▁€ ▁™ ▁s ▁ not ▁ a ▁ good ▁ thing ▁ to ▁ do ▁. ▁ We ▁ should ▁ focus ▁ on ▁ the ▁ math ▁ lesson ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁The ▁ clever ▁ child ▁ laughed ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Don ▁â ▁€ ▁™ ▁t ▁ worry ▁, ▁ I ▁ just ▁ thought ▁ it ▁ would ▁ be ▁ fun ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁Still ▁, ▁ the ▁ foolish ▁ one ▁ didn ▁â ▁€ ▁™ ▁t ▁ bud ▁ge ▁. ▁ He ▁ said ▁, ▁ � ▁� █€ ▁� ▁� ▁No ▁, ▁ let ▁â ▁€ ▁™ ▁s ▁ go ▁ to ▁ school ▁ now ▁. ▁ Math ▁ is ▁ more ▁ important ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁So ▁ the ▁ two ▁ children ▁ continued ▁ on ▁ their ▁ way ▁ to ▁ school ▁. ▁ But ▁ when ▁ they ▁ got ▁ there ▁, ▁ they ▁ were ▁ surprised ▁ to ▁ see ▁ their ▁ teacher ▁ with ▁ a ▁ garden ▁ fork ▁ in ▁ his ▁ hand ▁. ▁ He ▁ was ▁ mad ▁ and ▁ said ▁ to ▁ the ▁ two ▁ children ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Why ▁ did ▁ you ▁ poke ▁ my ▁ garden ▁?! ▁â ▁€ ▁  ▁\\n ▁\\n ▁The ▁ clever ▁ child ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁I ▁ told ▁ the ▁ foolish ▁ one ▁ that ▁ it ▁ would ▁ be ▁ fun ▁. ▁ He ▁ didn ▁â ▁€ ▁™ ▁t ▁ want ▁ to ▁ do ▁ it ▁, ▁ so ▁ I ▁ did ▁ it ▁ myself ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁The ▁ teacher ▁ shook ▁ his ▁ head ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁That ▁ was ▁ foolish ▁ of ▁ you ▁. ▁ Next ▁ time ▁, ▁ focus ▁ on ▁ math ▁ instead ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁The ▁ two ▁ children ▁ learned ▁ their ▁ lesson ▁

</details>

Example 10: `Anna ▁ stepped ▁ forward ▁ and ▁ said ▁ in ▁ a ▁ friendly ▁ voice ▁, ▁ � ▁� █€ ▁� ▁� ▁Hello ▁ there ▁!`

<details>
<summary>Click to see all of example 10</summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ an ▁ ordinary ▁ house ▁. ▁ Inside ▁ the ▁ house ▁ lived ▁ two ▁ children ▁, ▁ Mike ▁ and ▁ Anna ▁. ▁\\n ▁\\n ▁One ▁ day ▁, ▁ Mike ▁ said ▁ to ▁ Anna ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Let ▁â ▁€ ▁™ ▁s ▁ go ▁ on ▁ an ▁ adventure ▁. ▁ Let ▁â ▁€ ▁™ ▁s ▁ bring ▁ our ▁ toys ▁ and ▁ explore ▁ the ▁ wild ▁! ▁â ▁€ ▁ Anna ▁ smiled ▁ and ▁ happily ▁ agreed ▁ to ▁ go ▁ on ▁ an ▁ adventure ▁. ▁\\n ▁\\n ▁So ▁, ▁ Mike ▁ and ▁ Anna ▁ grabbed ▁ their ▁ toys ▁ and ▁ went ▁ outside ▁. ▁  ▁\\n ▁\\n ▁The ▁ wild ▁ was ▁ filled ▁ with ▁ tall ▁ trees ▁, ▁ colourful ▁ flowers ▁ and ▁ butterflies ▁ flying ▁ around ▁. ▁ It ▁ was ▁ a ▁ beautiful ▁ sight ▁. ▁ Anna ▁ gigg ▁led ▁ with ▁ delight ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Look ▁, ▁ Mike ▁! ▁â ▁€ ▁\\n ▁\\n ▁Eventually ▁, ▁ Mike ▁ and ▁ Anna ▁ noticed ▁ a ▁ tiny ▁, ▁ furry ▁ creature ▁. ▁ It ▁ was ▁ the ▁ cut ▁est ▁ thing ▁ they ▁ had ▁ ever ▁ seen ▁. ▁ Mike ▁ said ▁ gently ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Let ▁â ▁€ ▁™ ▁s ▁ introduce ▁ ourselves ▁. ▁ Don ▁â ▁€ ▁™ ▁t ▁ be ▁ scared ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁Anna ▁ stepped ▁ forward ▁ and ▁ said ▁ in ▁ a ▁ friendly ▁ voice ▁, ▁ � ▁� █€ ▁� ▁� ▁Hello ▁ there ▁! ▁ We ▁â ▁€ ▁™ ▁re ▁ Mike ▁ and ▁ Anna ▁. ▁ It ▁â ▁€ ▁™ ▁s ▁ nice ▁ to ▁ meet ▁ you ▁! ▁â ▁€ ▁\\n ▁\\n ▁The ▁ tiny ▁ creature ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� ▇€ ▁� ▁� ▁Hi ▁ there ▁! ▁ My ▁ name ▁ is ▁ Brown ▁ie ▁. ▁ Welcome ▁ to ▁ the ▁ wild ▁! ▁â ▁€ ▁  ▁\\n ▁\\n ▁Mike ▁ and ▁ Anna ▁ smiled ▁ and ▁ they ▁ all ▁ became ▁ great ▁ friends ▁. ▁ From ▁ that ▁ day ▁ onward ▁, ▁ they ▁ would ▁ often ▁ explore ▁ the ▁ wild ▁ together ▁. ▁

</details>

### Feature 1

My interpretation: This feature activates on a list of concrete nouns,
almost always in the context of characters wondering about this list or playing pretend with the list.

GPT-4o's interpretation: "Strong highlights for imaginary roles or objects."

Example 1: `He ▁ pret ▁ends ▁ it ▁ is ▁ a ▂ plane ▂ or ▃ a █ bird ▆ or ▄ a ▇ rocket ▅. ▁`

<details>
<summary>Click to see all of example 1</summary>

> Tom ▁ and ▁ Anna ▁ are ▁ playing ▁ in ▁ the ▁ backyard ▁. ▁ They ▁ see ▁ a ▁ big ▁ wire ▁ on ▁ the ▁ ground ▁. ▁ It ▁ is ▁ long ▁ and ▁ shiny ▁ and ▁ has ▁ a ▁ hook ▁ at ▁ the ▁ end ▁. ▁\\n ▁\\n ▁" ▁Look ▁, ▁ a ▁ wire ▁!" ▁ Tom ▁ says ▁. ▁ " ▁Let ▁'s ▁ play ▁ with ▁ it ▁!" ▁\\n ▁\\n ▁" ▁OK ▁!" ▁ Anna ▁ says ▁. ▁ " ▁What ▁ can ▁ we ▁ do ▁ with ▁ it ▁?" ▁\\n ▁\\n ▁Tom ▁ thinks ▁ for ▁ a ▁ moment ▁. ▁ He ▁ sees ▁ a ▁ big ▁ rock ▁ near ▁ the ▁ fence ▁. ▁ He ▁ has ▁ an ▁ idea ▁. ▁\\n ▁\\n ▁" ▁Let ▁'s ▁ weigh ▁ the ▁ rock ▁!" ▁ he ▁ says ▁. ▁ " ▁We ▁ can ▁ use ▁ the ▁ wire ▁ as ▁ a ▁ scale ▁!" ▁\\n ▁\\n ▁He ▁ picks ▁ up ▁ the ▁ wire ▁ and ▁ wraps ▁ it ▁ around ▁ the ▁ rock ▁. ▁ He ▁ holds ▁ the ▁ hook ▁ in ▁ his ▁ hand ▁ and ▁ lifts ▁ the ▁ rock ▁. ▁\\n ▁\\n ▁" ▁Wow ▁, ▁ it ▁'s ▁ heavy ▁!" ▁ he ▁ says ▁. ▁ " ▁Can ▁ you ▁ guess ▁ how ▁ much ▁ it ▁ weighs ▁?" ▁\\n ▁\\n ▁Anna ▁ looks ▁ at ▁ the ▁ rock ▁. ▁ She ▁ does ▁ not ▁ know ▁ how ▁ to ▁ weigh ▁ things ▁. ▁ She ▁ does ▁ not ▁ know ▁ what ▁ a ▁ scale ▁ is ▁. ▁ She ▁ does ▁ not ▁ know ▁ what ▁ a ▁ pound ▁ or ▁ a ▃ kil ▁o ▁ is ▁. ▁ She ▁ only ▁ knows ▁ big ▁ and ▁ small ▁, ▁ light ▁ and ▁ heavy ▁. ▁\\n ▁\\n ▁" ▁I ▁ don ▁'t ▁ know ▁," ▁ she ▁ says ▁. ▁ " ▁Maybe ▁ it ▁ weighs ▁... ▁ a ▁ lot ▁?" ▁\\n ▁\\n ▁Tom ▁ laughs ▁. ▁ He ▁ thinks ▁ Anna ▁ is ▁ funny ▁. ▁ He ▁ likes ▁ playing ▁ with ▁ her ▁. ▁\\n ▁\\n ▁" ▁Maybe ▁ it ▁ does ▁," ▁ he ▁ says ▁. ▁ " ▁But ▁ I ▁ have ▁ a ▁ better ▁ idea ▁. ▁ Let ▁'s ▁ swing ▁ the ▁ rock ▁!" ▁\\n ▁\\n ▁He ▁ swings ▁ the ▁ rock ▁ with ▁ the ▁ wire ▁. ▁ He ▁ makes ▁ it ▁ go ▁ up ▁ and ▁ down ▁, ▂ left ▂ and ▃ right ▄. ▁ He ▁ pret ▁ends ▁ it ▁ is ▁ a ▂ plane ▂ or ▃ a █ bird ▆ or ▄ a ▇ rocket ▅. ▁\\n ▁\\n ▁Anna ▁ cl ▁aps ▁ her ▁ hands ▁. ▁ She ▁ thinks ▁ Tom ▁ is ▁ clever ▁. ▁ She ▁ likes ▁ swinging ▁ the ▁ rock ▁. ▁\\n ▁\\n ▁" ▁Y ▁ay ▁, ▁ this ▁ is ▁ fun ▁!" ▁ she ▁ says ▁. ▁ " ▁Let ▁'s ▁ swing ▁ it ▁ higher ▁!" ▁\\n ▁\\n ▁They ▁ swing ▁ the ▁ rock ▁ higher ▁ and ▁ higher ▁. ▁ They ▁ do ▁ not ▁ see ▁ the ▁ unknown ▁ man ▁ who ▁ is ▁ watching ▁ them ▁ from ▁ the ▁ other ▁ side ▁ of ▁ the ▁ fence ▁. ▁ He ▁ is ▁ angry ▁. ▁ He ▁ does ▁ not ▁ like ▁ children ▁. ▁ He ▁ does ▁ not ▁ like ▁ noise ▁. ▁ He ▁ does ▁ not ▁ like ▁ his ▁ wire ▁ being ▁ used ▁ as ▁ a ▁ toy ▂. ▁\\n ▁\\n ▁He ▁ shouts ▁ at ▁ them ▁. ▁ He ▁ tells ▁ them ▁ to ▁ stop ▁. ▁ He ▁ tells ▁ them ▁ to ▁ give ▁ him ▁ back ▁ his ▁ wire ▁. ▁ He ▁ tells ▁ them ▁ to ▁ go ▁ away ▁. ▁\\n ▁\\n ▁Tom ▁ and ▁ Anna ▁ are ▁ scared ▁. ▁ They ▁ do ▁ not ▁ know ▁ the ▁ man ▁. ▁ They ▁ do ▁ not ▁ know ▁ why ▁ he ▁ is ▁ mad ▁. ▁ They ▁ do ▁ not ▁ know ▁ what ▁ to ▁ do ▁. ▁\\n ▁\\n ▁They ▁ drop ▁ the ▁ wire ▁ and ▁ the ▁ rock ▁. ▁ They ▁ run ▁ to ▁ the ▁ house ▁. ▁ They ▁ tell ▁ their ▁ mom ▁ what ▁ happened ▁. ▁\\n ▁\\n ▁Their ▁ mom ▁ hugs ▁ them ▁. ▁ She ▁ tells ▁ them ▁ they ▁ are ▁ safe ▁. ▁ She ▁ tells ▁ them ▁ they ▁ did ▁ nothing ▁ wrong ▁. ▁ She ▁ tells ▁ them ▁ the ▁ wire ▁ belongs ▁ to ▁ the ▁ electric ▁ company ▁. ▁ She ▁ tells ▁ them ▁ the ▁ man ▁ is ▁ a ▁ worker ▁ who ▁ is ▁ fixing ▁ the ▁ power ▁ lines ▁. ▁ She ▁ tells ▁ them ▁ he ▁ should ▁ not ▁ have ▁ yelled ▁ at ▁ them ▁. ▁ She ▁ tells ▁ them ▁ he ▁ should ▁ have ▁ been ▁ nicer ▁. ▁\\n ▁\\n ▁Tom ▁ and ▁ Anna ▁ feel ▁ better ▁. ▁ They ▁ understand ▁ now ▁. ▁ They ▁ are ▁ not ▁ afraid ▁ anymore ▁. ▁\\n ▁\\n ▁They ▁ go ▁ back ▁ to ▁ the ▁ backyard ▁. ▁ They ▁ find ▁ a ▁ new ▁ toy ▁ to ▁ play ▁ with ▁. ▁ They ▁ have ▁ fun ▁. ▁ They ▁ are ▁ happy ▁. ▁

</details>

Example 2: `and ▁ pretend ▁ it ▁ was ▁ a ▂ spaceship ▃ or ▅ a █ tunnel ▅. ▁`

<details>
<summary>Click to see all of example 2</summary>

> Tom ▁ and ▁ Lily ▁ were ▁ playing ▁ with ▁ a ▁ big ▁ tube ▁ in ▁ the ▁ garden ▁. ▁ The ▁ tube ▁ was ▁ white ▁ and ▁ had ▁ a ▁ red ▁ cap ▁. ▁ They ▁ liked ▁ to ▁ crawl ▁ inside ▁ the ▁ tube ▁ and ▁ pretend ▁ it ▁ was ▁ a ▂ spaceship ▃ or ▅ a █ tunnel ▅. ▁\\n ▁\\n ▁" ▁Let ▁'s ▁ go ▁ to ▁ the ▁ moon ▁!" ▁ Tom ▁ said ▁, ▁ putting ▁ the ▁ cap ▁ on ▁ one ▁ end ▁ of ▁ the ▁ tube ▁. ▁\\n ▁\\n ▁" ▁OK ▁!" ▁ Lily ▁ said ▁, ▁ following ▁ him ▁ inside ▁. ▁ They ▁ made ▁ noises ▁ like ▁ rockets ▁ and ▁ stars ▁. ▁\\n ▁\\n ▁But ▁ they ▁ did ▁ not ▁ know ▁ that ▁ the ▁ tube ▁ was ▁ not ▁ a ▁ toy ▁. ▁ It ▁ was ▁ a ▁ special ▁ tube ▁ that ▁ their ▁ dad ▁ used ▁ for ▁ his ▁ work ▁. ▁ The ▁ tube ▁ had ▁ a ▁ button ▁ that ▁ could ▁ make ▁ things ▁ shrink ▁ or ▁ grow ▁. ▁\\n ▁\\n ▁When ▁ Tom ▁ and ▁ Lily ▁ were ▁ inside ▁, ▁ the ▁ button ▁ got ▁ pressed ▁ by ▁ a ▁ rock ▁. ▁ The ▁ tube ▁ started ▁ to ▁ make ▁ a ▁ loud ▁ sound ▁ and ▁ a ▁ bright ▁ light ▁. ▁ Tom ▁ and ▁ Lily ▁ felt ▁ very ▁ scared ▁ and ▁ tried ▁ to ▁ get ▁ out ▁. ▁\\n ▁\\n ▁" ▁Help ▁! ▁ Help ▁!" ▁ they ▁ shouted ▁. ▁ " ▁We ▁ are ▁ stuck ▁!" ▁\\n ▁\\n ▁Their ▁ dad ▁ heard ▁ them ▁ and ▁ ran ▁ to ▁ the ▁ garden ▁. ▁ He ▁ saw ▁ the ▁ tube ▁ and ▁ realized ▁ what ▁ had ▁ happened ▁. ▁ He ▁ tried ▁ to ▁ press ▁ the ▁ button ▁ again ▁, ▁ but ▁ it ▁ was ▁ broken ▁. ▁ He ▁ could ▁ not ▁ make ▁ the ▁ tube ▁ stop ▁. ▁\\n ▁\\n ▁He ▁ looked ▁ inside ▁ the ▁ tube ▁ and ▁ saw ▁ Tom ▁ and ▁ Lily ▁. ▁ They ▁ were ▁ very ▁ small ▁, ▁ like ▁ ants ▁. ▁ They ▁ looked ▁ at ▁ him ▁ with ▁ big ▁ eyes ▁ and ▁ cried ▁. ▁\\n ▁\\n ▁" ▁Dad ▁! ▁ Dad ▁! ▁ Save ▁ us ▁!" ▁ they ▁ said ▁. ▁\\n ▁\\n ▁But ▁ their ▁ dad ▁ could ▁ not ▁ hear ▁ them ▁. ▁ He ▁ could ▁ not ▁ reach ▁ them ▁. ▁ He ▁ could ▁ not ▁ help ▁ them ▁. ▁\\n ▁\\n ▁He ▁ felt ▁ very ▁ sad ▁ and ▁ sorry ▁. ▁ He ▁ had ▁ lost ▁ his ▁ children ▁. ▁ They ▁ had ▁ shrunk ▁ in ▁ the ▁ tube ▁. ▁

</details>

Example 3: `and ▁ pretend ▁ it ▁ is ▁ a ▂ spaceship ▃, ▃ a ▆ castle ▄ or ▆ a █ dragon ▄. ▁`

<details>
<summary>Click to see all of example 3</summary>

> L ▁ily ▁ and ▁ Max ▁ are ▁ best ▁ friends ▁. ▁ They ▁ like ▁ to ▁ play ▁ in ▁ the ▁ park ▁ every ▁ day ▁. ▁ They ▁ have ▁ a ▁ lot ▁ of ▁ fun ▁ on ▁ the ▁ swings ▁, ▁ the ▁ slide ▁ and ▁ the ▃ sees ▁aw ▁. ▁ But ▁ their ▁ favorite ▁ thing ▁ is ▁ the ▁ big ▁ rock ▁. ▁\\n ▁\\n ▁The ▁ big ▁ rock ▁ is ▁ incredible ▁. ▁ It ▁ is ▁ very ▁ big ▁ and ▁ very ▁ smooth ▁. ▁ It ▁ has ▁ many ▁ colors ▁ and ▁ shapes ▁ on ▁ it ▁. ▁ Lily ▁ and ▁ Max ▁ like ▁ to ▁ climb ▁ on ▁ it ▁, ▁ sit ▁ on ▁ it ▁ and ▁ pretend ▁ it ▁ is ▁ a ▂ spaceship ▃, ▃ a ▆ castle ▄ or ▆ a █ dragon ▄. ▁\\n ▁\\n ▁One ▁ day ▁, ▁ they ▁ find ▁ a ▁ new ▁ game ▁. ▁ They ▁ sit ▁ on ▁ the ▁ rock ▁ and ▁ hold ▁ hands ▁. ▁ Then ▁ they ▁ lean ▁ back ▁ and ▁ forth ▁, ▁ making ▁ the ▁ rock ▁ move ▁ a ▁ little ▁. ▁ They ▁ g ▁iggle ▁ and ▁ shout ▁, ▁ " ▁We ▁ are ▁ rocking ▁ the ▁ rock ▁! ▁ We ▁ are ▁ rocking ▁ the ▁ rock ▁!" ▁\\n ▁\\n ▁They ▁ rock ▁ the ▁ rock ▁ faster ▁ and ▁ faster ▁, ▁ feeling ▁ the ▁ wind ▁ in ▁ their ▁ hair ▁ and ▁ the ▁ sun ▁ on ▁ their ▁ faces ▁. ▁ They ▁ are ▁ very ▁ happy ▁ and ▁ their ▁ smiles ▁ are ▁ very ▁ big ▁. ▁\\n ▁\\n ▁" ▁Look ▁ at ▁ us ▁!" ▁ Lily ▁ says ▁. ▁ " ▁We ▁ are ▁ the ▁ best ▁ rock ▁ers ▁ in ▁ the ▁ park ▁!" ▁\\n ▁\\n ▁" ▁Yes ▁, ▁ we ▁ are ▁!" ▁ Max ▁ says ▁. ▁ " ▁We ▁ are ▁ the ▁ best ▁ friends ▁ in ▁ the ▁ world ▁!" ▁\\n ▁\\n ▁They ▁ rock ▁ the ▁ rock ▁ until ▁ they ▁ are ▁ tired ▁. ▁ Then ▁ they ▁ hug ▁ each ▁ other ▁ and ▁ say ▁, ▁ " ▁Thank ▁ you ▁ for ▁ rocking ▁ with ▁ me ▁. ▁ You ▁ are ▁ incredible ▁!" ▁\\n ▁\\n ▁They ▁ get ▁ off ▁ the ▁ rock ▁ and ▁ run ▁ to ▁ their ▁ moms ▁, ▁ who ▁ are ▁ waiting ▁ for ▁ them ▁ with ▁ juice ▁ and ▁ cookies ▂. ▁ They ▁ tell ▁ them ▁ about ▁ their ▁ game ▁ and ▁ how ▁ much ▁ fun ▁ they ▁ had ▁. ▁ Their ▁ moms ▁ smile ▁ and ▁ say ▁, ▁ " ▁We ▁ are ▁ glad ▁ you ▁ had ▁ a ▁ good ▁ time ▁. ▁ You ▁ are ▁ incredible ▁, ▁ too ▁!" ▁

</details>

Example 4: `Sometimes ▁ they ▁ pretended ▁ it ▁ was ▁ a ▃ fire ▂ truck ▁ or ▄ a █ garbage ▅ truck ▄ or ▆ a ▇ delivery ▄ truck ▄. ▁`

<details>
<summary>Click to see all of example 4</summary>

> Ben ▁ and ▁ Lily ▁ liked ▁ to ▁ play ▁ with ▁ their ▁ toy ▁ truck ▁. ▁ It ▁ was ▁ big ▁ and ▁ red ▁ and ▁ had ▁ four ▁ wheels ▁. ▁ They ▁ could ▁ push ▁ it ▁ around ▁ the ▁ yard ▁ and ▁ make ▁ v ▁room ▁- ▁v ▁room ▁ noises ▁. ▁ Sometimes ▁ they ▁ pretended ▁ it ▁ was ▁ a ▃ fire ▂ truck ▁ or ▄ a █ garbage ▅ truck ▄ or ▆ a ▇ delivery ▄ truck ▄. ▁\\n ▁\\n ▁One ▁ day ▁, ▁ they ▁ decided ▁ to ▁ play ▁ delivery ▁ truck ▁. ▁ They ▁ looked ▁ for ▁ things ▁ to ▁ load ▁ in ▁ the ▁ back ▁ of ▁ the ▁ truck ▁. ▁ They ▁ found ▁ some ▁ rocks ▁, ▁ some ▁ leaves ▁, ▁ some ▁ sticks ▁ and ▁ some ▁ flowers ▁. ▁ They ▁ put ▁ them ▁ in ▁ the ▁ truck ▁ and ▁ said ▁, ▁ " ▁This ▁ is ▁ our ▁ cargo ▁. ▁ We ▁ have ▁ to ▁ deliver ▁ it ▁ to ▁ our ▁ customers ▁." ▁\\n ▁\\n ▁They ▁ drove ▁ the ▁ truck ▁ around ▁ the ▁ yard ▁, ▁ stopping ▁ at ▁ different ▁ places ▁. ▁ They ▁ gave ▁ a ▁ rock ▁ to ▁ the ▂ dog ▁, ▂ a ▃ leaf ▃ to ▁ the ▃ bird ▁, ▃ a ▄ stick ▂ to ▁ the ▂ cat ▂ and ▂ a ▄ flower ▃ to ▁ the ▄ mom ▁. ▁ They ▁ said ▁, ▁ " ▁Here ▁ is ▁ your ▁ cargo ▁. ▁ Thank ▁ you ▁ for ▁ choosing ▁ our ▁ delivery ▁ truck ▁." ▁\\n ▁\\n ▁Then ▁ they ▁ saw ▁ a ▁ bunch ▁ of ▁ bananas ▁ on ▁ the ▁ kitchen ▁ table ▁. ▁ They ▁ thought ▁, ▁ " ▁Ban ▁anas ▁ are ▁ y ▁ummy ▁. ▁ We ▁ want ▁ some ▁ bananas ▁." ▁ They ▁ ran ▁ to ▁ the ▁ table ▁ and ▁ grabbed ▁ the ▁ bananas ▁. ▁ They ▁ put ▁ them ▁ in ▁ the ▁ truck ▁ and ▁ said ▁, ▁ " ▁This ▁ is ▁ our ▁ special ▁ cargo ▁. ▁ We ▁ have ▁ to ▁ deliver ▁ it ▁ to ▁ ourselves ▁." ▁\\n ▁\\n ▁They ▁ drove ▁ the ▁ truck ▁ to ▁ their ▁ favorite ▁ spot ▁ under ▁ the ▁ tree ▁. ▁ They ▁ unloaded ▁ the ▁ bananas ▁ and ▁ peeled ▁ them ▁. ▁ They ▁ ate ▁ them ▁ and ▁ said ▁, ▁ " ▁M ▁mm ▁, ▁ these ▁ are ▁ the ▁ best ▁ bananas ▁ ever ▁. ▁ We ▁ are ▁ mighty ▁ delivery ▁ truck ▁ drivers ▁. ▁ We ▁ can ▁ load ▁ and ▁ un ▁load ▁ anything ▁." ▁ They ▁ smiled ▁ and ▁ hugged ▁ each ▁ other ▁. ▁ They ▁ were ▁ happy ▁ and ▁ full ▁. ▁

</details>

Example 5: `She ▁ pret ▁ends ▁ she ▁ is ▁ a ▂ mouse ▁ or ▃ a █ cat ▅ or ▅ a ▇ snake ▄. ▁`

<details>
<summary>Click to see all of example 5</summary>

> Anna ▁ likes ▁ to ▁ crawl ▁. ▁ She ▁ craw ▁ls ▁ under ▁ the ▁ table ▁, ▁ under ▁ the ▁ sofa ▁, ▂ under ▁ the ▁ bed ▁. ▁ She ▁ pret ▁ends ▁ she ▁ is ▁ a ▂ mouse ▁ or ▃ a █ cat ▅ or ▅ a ▇ snake ▄. ▁ She ▁ makes ▁ funny ▁ noises ▁ and ▁ gigg ▁les ▁. ▁\\n ▁\\n ▁One ▁ day ▁, ▁ she ▁ craw ▁ls ▁ under ▁ the ▁ window ▁ and ▁ sees ▁ a ▁ big ▁ aer ▁opl ▁ane ▁ in ▁ the ▁ sky ▁. ▁ It ▁ is ▁ white ▁ and ▁ shiny ▁ and ▁ has ▁ wings ▁ and ▁ a ▁ tail ▁. ▁ It ▁ makes ▁ a ▁ loud ▁ noise ▁ and ▁ flies ▁ very ▁ fast ▁. ▁ Anna ▁ is ▁ amazed ▁ and ▁ excited ▁. ▁ She ▁ wants ▁ to ▁ see ▁ more ▁ aer ▁opl ▁anes ▁. ▁\\n ▁\\n ▁She ▁ craw ▁ls ▁ out ▁ of ▁ the ▁ window ▁ and ▁ into ▁ the ▁ garden ▁. ▁ She ▁ looks ▁ up ▁ and ▁ sees ▁ many ▁ aer ▁opl ▁anes ▁. ▁ Some ▁ are ▁ big ▁, ▁ some ▁ are ▁ small ▁, ▁ some ▁ are ▁ red ▁, ▁ some ▁ are ▂ blue ▂. ▁ They ▁ fly ▁ in ▁ different ▁ directions ▁ and ▁ make ▁ different ▁ noises ▁. ▁ Anna ▁ thinks ▁ they ▁ are ▁ busy ▁. ▁ They ▁ must ▁ have ▁ many ▁ places ▁ to ▁ go ▁ and ▁ many ▁ things ▁ to ▁ do ▁. ▁\\n ▁\\n ▁She ▁ craw ▁ls ▁ to ▁ the ▁ fence ▁ and ▁ waves ▁ at ▁ the ▁ aer ▁opl ▁anes ▁. ▁ She ▁ hopes ▁ they ▁ see ▁ her ▁ and ▁ wave ▁ back ▁. ▁ She ▁ says ▁ hello ▁ and ▁ goodbye ▁ and ▁ thank ▁ you ▁ and ▁ please ▁. ▁ She ▁ thinks ▁ they ▁ are ▁ friendly ▁ and ▁ nice ▁. ▁\\n ▁\\n ▁She ▁ craw ▁ls ▁ back ▁ to ▁ the ▁ window ▁ and ▁ sees ▁ her ▁ mum ▁. ▁ Her ▁ mum ▁ smiles ▁ and ▁ picks ▁ her ▁ up ▁. ▁ She ▁ hugs ▁ her ▁ and ▁ kisses ▁ her ▁ and ▁ says ▁ she ▁ loves ▁ her ▁. ▁ She ▁ asks ▁ her ▁ what ▁ she ▁ was ▁ doing ▁. ▁ Anna ▁ tells ▁ her ▁ she ▁ was ▁ crawling ▁ and ▁ watching ▁ the ▁ aer ▁opl ▁anes ▁. ▁ She ▁ says ▁ they ▁ are ▁ busy ▁ and ▁ friendly ▁ and ▁ nice ▁. ▁ Her ▁ mum ▁ says ▁ she ▁ is ▁ clever ▁ and ▁ curious ▁ and ▁ brave ▁. ▁ She ▁ says ▁ they ▁ can ▁ watch ▁ the ▁ aer ▁opl ▁anes ▁ together ▁. ▁ Anna ▁ is ▁ happy ▁ and ▁ proud ▁. ▁ She ▁ loves ▁ her ▁ mum ▁ and ▁ the ▁ aer ▁opl ▁anes ▁. ▁

</details>

Example 6: `They ▁ pretended ▁ they ▁ were ▁ crossing ▁ a ▁ jungle ▁ or ▅ a █ mountain ▅.`

<details>
<summary>Click to see all of example 6</summary>

> L ▁ily ▁ and ▁ Ben ▁ were ▁ best ▁ friends ▁. ▁ They ▁ liked ▁ to ▁ play ▁ in ▁ the ▁ park ▁ every ▁ day ▁. ▁ They ▁ had ▁ a ▁ lot ▁ of ▁ fun ▁ on ▁ the ▁ swings ▁, ▁ the ▁ slide ▁ and ▁ the ▃ sees ▁aw ▁. ▁ But ▁ their ▁ favorite ▁ thing ▁ was ▁ the ▁ rope ▁ bridge ▁. ▁ It ▁ was ▁ high ▁ and ▁ long ▁ and ▁ w ▁obb ▁ly ▁. ▁ They ▁ pretended ▁ they ▁ were ▁ crossing ▁ a ▁ jungle ▁ or ▅ a █ mountain ▅. ▁\\n ▁\\n ▁One ▁ day ▁, ▁ they ▁ saw ▁ a ▁ new ▁ boy ▁ in ▁ the ▁ park ▁. ▁ His ▁ name ▁ was ▁ Max ▁. ▁ He ▁ looked ▁ sad ▁ and ▁ lonely ▁. ▁ Lily ▁ and ▁ Ben ▁ wanted ▁ to ▁ be ▁ nice ▁ to ▁ him ▁. ▁ They ▁ asked ▁ him ▁ if ▁ he ▁ wanted ▁ to ▁ play ▁ with ▁ them ▁. ▁ Max ▁ nodded ▁ and ▁ smiled ▁. ▁ They ▁ showed ▁ him ▁ the ▁ rope ▁ bridge ▁ and ▁ told ▁ him ▁ it ▁ was ▁ fun ▁. ▁\\n ▁\\n ▁But ▁ Max ▁ was ▁ scared ▁ of ▁ the ▁ rope ▁ bridge ▁. ▁ He ▁ thought ▁ it ▁ was ▁ too ▁ high ▁ and ▁ too ▁ w ▁obb ▁ly ▁. ▁ He ▁ did ▁ not ▁ want ▁ to ▁ cross ▁ it ▁. ▁ He ▁ said ▁ he ▁ would ▁ wait ▁ for ▁ them ▁ on ▁ the ▁ other ▁ side ▁. ▁ Lily ▁ and ▁ Ben ▁ said ▁ okay ▁ and ▁ started ▁ to ▁ cross ▁ the ▁ rope ▁ bridge ▁. ▁ They ▁ held ▁ the ▁ rope ▁ tight ▁ and ▁ walked ▁ slowly ▁. ▁\\n ▁\\n ▁But ▁ when ▁ they ▁ were ▁ in ▁ the ▁ middle ▁ of ▁ the ▁ bridge ▁, ▁ they ▁ heard ▁ a ▁ loud ▁ snap ▁. ▁ One ▁ of ▁ the ▁ ropes ▁ had ▁ broken ▁. ▁ The ▁ bridge ▁ tilted ▁ and ▁ shook ▁. ▁ Lily ▁ and ▁ Ben ▁ screamed ▁. ▁ They ▁ were ▁ afraid ▁ they ▁ would ▁ fall ▁. ▁ They ▁ looked ▁ for ▁ Max ▁, ▁ but ▁ he ▁ was ▁ not ▁ there ▁. ▁ He ▁ had ▁ run ▁ away ▁. ▁\\n ▁\\n ▁L ▁ily ▁ and ▁ Ben ▁ did ▁ not ▁ know ▁ what ▁ to ▁ do ▁. ▁ They ▁ were ▁ stuck ▁ on ▁ the ▁ bridge ▁. ▁ They ▁ cried ▁ for ▁ help ▁. ▁ But ▁ no ▁ one ▁ heard ▁ them ▁. ▁ They ▁ wished ▁ Max ▁ had ▁ stayed ▁ with ▁ them ▁. ▁ They ▁ wished ▁ they ▁ had ▁ a ▁ friend ▁ who ▁ would ▁ not ▁ leave ▁ them ▁. ▁\\n ▁\\n ▁Then ▁ they ▁ saw ▁ a ▁ man ▁ coming ▁. ▁ He ▁ was ▁ wearing ▁ a ▁ hat ▁ and ▁ a ▁ badge ▁. ▁ He ▁ was ▁ a ▁ park ▁ ranger ▁. ▁ He ▁ saw ▁ Lily ▁ and ▁ Ben ▁ on ▁ the ▁ bridge ▁. ▁ He ▁ ran ▁ to ▁ help ▁ them ▁. ▁ He ▁ had ▁ a ▁ big ▁ rope ▁. ▁ He ▁ threw ▁ it ▁ to ▁ them ▁ and ▁ told ▁ them ▁ to ▁ hold ▁ it ▁ tight ▁. ▁ He ▁ pulled ▁ them ▁ to ▁ safety ▁. ▁ He ▁ hugged ▁ them ▁ and ▁ said ▁ they ▁ were ▁ brave ▁. ▁\\n ▁\\n ▁L ▁ily ▁ and ▁ Ben ▁ thanked ▁ the ▁ park ▁ ranger ▁. ▁ They ▁ were ▁ happy ▁ he ▁ had ▁ saved ▁ them ▁. ▁ They ▁ learned ▁ a ▁ lesson ▁ that ▁ day ▁. ▁ They ▁ learned ▁ that ▁ a ▁ true ▁ friend ▁ is ▁ someone ▁ who ▁ remains ▁ with ▁ you ▁ when ▁ you ▁ need ▁ them ▁. ▁ They ▁ learned ▁ that ▁ a ▁ rope ▁ can ▁ be ▁ tight ▁ or ▁ loose ▁, ▁ but ▁ a ▁ friendship ▁ should ▁ always ▁ be ▁ tight ▁. ▁

</details>

Example 7: `Sometimes ▁ they ▁ pretended ▁ that ▁ the ▁ curtain ▁ was ▁ a ▂ door ▂ to ▁ a ▂ different ▁ place ▁, ▂ like ▁ a ▂ castle ▂ or ▅ a █ jungle ▅ or ▆ a ▇ spaceship ▇. ▁`

<details>
<summary>Click to see all of example 7</summary>

> L ▁ily ▁ and ▁ Ben ▁ were ▁ twins ▁ who ▁ liked ▁ to ▁ play ▁ in ▁ their ▁ room ▁. ▁ They ▁ had ▁ a ▁ big ▁ curtain ▁ that ▁ hung ▁ from ▁ the ▁ ceiling ▁ and ▁ made ▁ a ▁ wall ▁ in ▁ the ▁ middle ▁ of ▁ the ▁ room ▁. ▁ Sometimes ▁ they ▁ pretended ▁ that ▁ the ▁ curtain ▁ was ▁ a ▂ door ▂ to ▁ a ▂ different ▁ place ▁, ▂ like ▁ a ▂ castle ▂ or ▅ a █ jungle ▅ or ▆ a ▇ spaceship ▇. ▁\\n ▁\\n ▁One ▁ day ▁, ▁ they ▁ decided ▁ to ▁ play ▁ a ▁ game ▁ of ▁ hide ▁ and ▁ seek ▁. ▁ Lily ▁ said ▁, ▁ " ▁I ▁ will ▁ count ▁ to ▁ ten ▁ and ▁ you ▁ hide ▁ behind ▁ the ▁ curtain ▁. ▁ Then ▁ I ▁ will ▁ try ▁ to ▁ find ▁ you ▁." ▁ Ben ▁ nodded ▁ and ▁ ran ▁ behind ▁ the ▁ curtain ▁. ▁ Lily ▁ closed ▁ her ▁ eyes ▁ and ▁ counted ▁ out ▁ loud ▁, ▁ " ▁One ▁, ▃ two ▁, ▂ three ▁, ▁ four ▁, ▁ five ▁, ▁ six ▁, ▁ seven ▁, ▁ eight ▁, ▁ nine ▁, ▁ ten ▁! ▁ Ready ▁ or ▁ not ▁, ▁ here ▁ I ▁ come ▁!" ▁\\n ▁\\n ▁L ▁ily ▁ opened ▁ her ▁ eyes ▁ and ▁ looked ▁ around ▁ the ▁ room ▁. ▁ She ▁ saw ▁ a ▁ small ▁ bump ▁ under ▁ the ▁ bed ▁, ▂ a ▂ pile ▁ of ▁ toys ▁ in ▃ the ▂ corner ▁, ▂ and ▂ a ▄ lamp ▁ on ▃ the ▁ table ▂. ▁ She ▁ thought ▁, ▁ " ▁Where ▁ is ▁ Ben ▁? ▁ Maybe ▁ he ▁ is ▁ under ▁ the ▁ bed ▁." ▁ She ▁ walked ▁ to ▁ the ▁ bed ▁ and ▁ lifted ▁ the ▁ blanket ▁. ▁ But ▁ there ▁ was ▁ no ▁ Ben ▁, ▁ only ▁ a ▁ stuffed ▁ bear ▁. ▁ She ▁ said ▁, ▁ " ▁O ▁ops ▁, ▁ not ▁ here ▁!" ▁\\n ▁\\n ▁She ▁ moved ▁ to ▁ the ▁ corner ▁ and ▁ looked ▁ at ▁ the ▁ toys ▁. ▁ She ▁ saw ▁ a ▁ truck ▁, ▂ a ▄ ball ▂, ▄ a ▄ doll ▃, ▄ and ▄ a ▄ book ▂. ▁ She ▁ thought ▁, ▁ " ▁Maybe ▁ he ▁ is ▁ behind ▁ the ▁ toys ▁." ▁ She ▁ pushed ▁ the ▁ toys ▁ aside ▁ and ▁ looked ▁ behind ▁ them ▁. ▁ But ▁ there ▁ was ▁ no ▁ Ben ▁, ▁ only ▁ a ▂ spider ▁ web ▁. ▁ She ▁ said ▁, ▁ " ▁Y ▁uck ▁, ▁ not ▁ here ▁!" ▁\\n ▁\\n ▁She ▁ walked ▁ to ▁ the ▁ table ▁ and ▁ looked ▁ at ▁ the ▁ lamp ▁. ▁ She ▁ saw ▁ a ▁ cord ▁, ▁ a ▃ switch ▁, ▃ a ▃ shade ▁, ▃ and ▃ a ▄ bulb ▁. ▁ She ▁ thought ▁, ▁ " ▁Maybe ▁ he ▁ is ▁ under ▁ the ▁ lamp ▁." ▁ She ▁ lifted ▁ the ▁ lamp ▁ and ▁ looked ▁ under ▁ it ▁. ▁ But ▁ there ▁ was ▁ no ▁ Ben ▁, ▁ only ▂ a ▂ dust ▁ bunny ▁. ▁ She ▁ said ▁, ▁ " ▁E ▁w ▁, ▁ not ▁ here ▁!" ▁\\n ▁\\n ▁She ▁ scratched ▁ her ▁ head ▁ and ▁ wondered ▁, ▁ " ▁Where ▁ is ▁ Ben ▁? ▁ He ▁ is ▁ very ▁ good ▁ at ▁ hiding ▁. ▁ Maybe ▁ he ▁ is ▁ behind ▁ the ▁ curtain ▁." ▁ She ▁ ran ▁ to ▁ the ▁ curtain ▁ and ▁ pulled ▁ it ▁ aside ▁. ▁ But ▁ there ▁ was ▁ no ▁ Ben ▁, ▁ only ▁ a ▂ big ▁ hole ▁. ▁ She ▁ gasped ▁ and ▁ said ▁, ▁ " ▁Oh ▁ no ▁, ▁ the ▁ curtain ▁ is ▁ split ▁!" ▁\\n ▁\\n ▁She ▁ heard ▁ a ▁ g ▁iggle ▁ from ▁ the ▁ other ▁ side ▁ of ▁ the ▁ hole ▁. ▁ She ▁ looked ▁ through ▁ it ▁ and ▁ saw ▁ Ben ▁. ▁ He ▁ was ▁ smiling ▁ and ▁ holding ▁ a ▁ pair ▁ of ▁ scissors ▁. ▁ He ▁ said ▁, ▁ " ▁Sur ▁prise ▁! ▁ I ▁ made ▁ a ▁ new ▁ door ▁. ▁ Do ▁ you ▁ like ▁ it ▁?" ▁ He ▁ was ▁ very ▁ playful ▁ and ▁ thought ▁ it ▁ was ▁ funny ▁. ▁\\n ▁\\n ▁L ▁ily ▁ was ▁ not ▁ amused ▁. ▁ She ▁ was ▁ angry ▁ and ▁ sad ▁. ▁ She ▁ said ▁, ▁ " ▁Ben ▁, ▁ that ▁ was ▁ not ▁ nice ▁. ▁ You ▁ ruined ▁ the ▁ curtain ▁. ▁ Mom ▁ and ▁ Dad ▁ will ▁ be ▁ mad ▁. ▁ How ▁ will ▁ we ▁ fix ▁ it ▁?" ▁ She ▁ started ▁ to ▁ cry ▁. ▁\\n ▁\\n ▁Ben ▁ realized ▁ that ▁ he ▁ had ▁ made ▁ a ▁ mistake ▁. ▁ He ▁ felt ▁ sorry ▁ and ▁ ashamed ▁. ▁ He ▁ said ▁, ▁ " ▁I ▁'m ▁ sorry ▁, ▁ Lily ▁. ▁ I ▁ didn ▁'t ▁ mean ▁ to ▁ make ▁ you ▁ cry ▁. ▁ I ▁ just ▁ wanted ▁ to ▁ have ▁ fun ▁. ▁ Please ▁ don ▁'t ▁ be ▁ mad ▁ at ▁ me ▁. ▁ Maybe ▁ we ▁ can ▁ sew ▁ the ▁ curtain ▁ back ▁ together ▁. ▁ Or ▁ maybe ▁ we ▁ can ▁ find ▁ a ▁ new ▁ curtain ▁. ▁ Or ▁ maybe ▁ we ▁ can ▁ pretend ▁ that ▁ the ▁ hole ▁ is ▁ a ▁ window ▁. ▁ Can ▁ we ▁ still ▁ be ▁ friends ▁?" ▁\\n ▁\\n ▁L ▁ily ▁ looked ▁ at ▁ Ben ▁ and ▁ saw ▁ that ▁ he ▁ was ▁ sincere ▁. ▁ She ▁ wiped ▁ her ▁ tears ▁ and ▁ said ▁, ▁ " ▁OK ▁, ▁ Ben ▁. ▁ I ▁ forgive ▁ you ▁. ▁ But ▁ you ▁ have ▁ to ▁ promise ▁ not ▁ to ▁ cut ▁ the ▁ curtain ▁ again ▁. ▁ And ▁ you ▁ have ▁ to ▁ help ▁ me ▁ clean ▁ up ▁ the ▁ mess ▁. ▁ And ▁ you ▁ have ▁ to ▁ let ▁ me ▁ hide ▁ next ▁ time ▁. ▁ Deal ▁?" ▁\\n ▁\\n ▁Ben ▁ nodded ▁ and ▁ said ▁, ▁ " ▁Deal ▁. ▁ I ▁'m ▁ sorry ▁, ▁ Lily ▁. ▁ I ▁ love ▁ you ▁." ▁ He ▁ hugged ▁ her ▁ and ▁ said ▁, ▁ " ▁You ▁ are ▁ the ▁ best ▁ sister ▁ ever ▁." ▁\\n ▁\\n ▁L ▁ily ▁ hugged ▁ him ▁ back ▁ and ▁ said ▁, ▁ " ▁You ▁ are ▁ the ▁ best ▁ brother ▁ ever ▁. ▁ But ▁ you ▁ are ▁ also ▁ very ▁ naughty ▁. ▁ Come ▁ on ▁, ▁ let ▁'s ▁ go ▁ find ▁ Mom ▁ and ▁ Dad ▁ and ▁ tell ▁ them ▁ what ▁ happened ▁. ▁ Maybe ▁ they ▁ will ▁ not ▁ be ▁ too ▁ mad ▁ if ▁ we ▁ say ▁ we ▁ are ▁ sorry ▁ and ▁ we ▁ will ▁ fix ▁ it ▁." ▁\\n ▁\\n ▁They ▁ held ▁ hands ▁ and ▁ walked ▁ out ▁ of ▁ the ▁ room ▁. ▁ They ▁ hoped ▁ that ▁ Mom ▁ and ▁ Dad ▁ would ▁ understand ▁. ▁ They ▁ learned ▁ that ▁ playing ▁ is ▁ fun ▁, ▁ but ▁ not ▁ when ▁ it ▁ hurts ▁ someone ▁ or ▁ something ▁. ▁ They ▁ also ▁ learned ▁ that ▁ saying ▁ sorry ▁ and ▁ forgiving ▁ are ▁ important ▁ when ▁ you ▁ make ▁ a ▁ mistake ▁. ▁ They ▁ were ▁ still ▁ twins ▁ who ▁ liked ▁ to ▁ play ▁, ▁ but ▁ they ▁ were ▁ also ▁ more ▁ careful ▁ and ▁ kind ▁. ▁

</details>

Example 8: `They ▁ can ▁ go ▁ inside ▁ and ▁ pretend ▁ it ▁ is ▁ a ▂ ship ▂ or ▄ a █ house ▆. ▁`

<details>
<summary>Click to see all of example 8</summary>

> L ▁ila ▁ and ▁ Tom ▁ are ▁ happy ▁. ▁ They ▁ like ▁ to ▁ play ▁ with ▁ their ▁ toys ▁ and ▁ books ▁. ▁ They ▁ have ▁ a ▁ big ▁ port ▁ in ▁ their ▁ room ▁. ▁ It ▁ is ▁ a ▁ big ▁ box ▁ with ▁ a ▁ hole ▁. ▁ They ▁ can ▁ go ▁ inside ▁ and ▁ pretend ▁ it ▁ is ▁ a ▂ ship ▂ or ▄ a █ house ▆. ▁\\n ▁\\n ▁One ▁ day ▁, ▁ L ▁ila ▁ has ▁ an ▁ idea ▁. ▁ She ▁ finds ▁ a ▁ big ▁ cloth ▁ in ▁ the ▁ closet ▁. ▁ She ▁ says ▁ to ▁ Tom ▁, ▁ " ▁Let ▁'s ▁ wrap ▁ the ▁ port ▁ with ▁ this ▁ cloth ▁. ▁ It ▁ will ▁ be ▁ a ▁ secret ▁ port ▁. ▁ No ▁ one ▁ can ▁ see ▁ us ▁ inside ▁." ▁\\n ▁\\n ▁Tom ▁ likes ▁ the ▁ idea ▁. ▁ He ▁ helps ▁ L ▁ila ▁ wrap ▁ the ▁ port ▁ with ▁ the ▁ cloth ▁. ▁ They ▁ t ▁uck ▁ the ▁ ends ▁ under ▁ the ▁ box ▁. ▁ They ▁ make ▁ sure ▁ there ▁ is ▁ a ▁ gap ▁ for ▁ the ▁ hole ▁. ▁ They ▁ crawl ▁ inside ▁ the ▁ port ▁. ▁ It ▁ is ▁ dark ▁ and ▁ cozy ▁. ▁\\n ▁\\n ▁They ▁ g ▁iggle ▁ and ▁ whisper ▁. ▁ They ▁ tell ▁ each ▁ other ▁ stories ▁ and ▁ jokes ▁. ▁ They ▁ play ▁ with ▁ their ▁ toys ▁ and ▁ books ▁. ▁ They ▁ have ▁ fun ▁ in ▁ their ▁ secret ▁ port ▁. ▁\\n ▁\\n ▁Mom ▁ comes ▁ to ▁ check ▁ on ▁ them ▁. ▁ She ▁ sees ▁ the ▁ wrapped ▁ port ▁ in ▁ their ▁ room ▁. ▁ She ▁ smiles ▁. ▁ She ▁ knows ▁ they ▁ are ▁ inside ▁. ▁ She ▁ says ▁, ▁ " ▁L ▁ila ▁, ▁ Tom ▁, ▁ are ▁ you ▁ in ▁ there ▁?" ▁\\n ▁\\n ▁L ▁ila ▁ and ▁ Tom ▁ hear ▁ Mom ▁. ▁ They ▁ say ▁, ▁ " ▁Yes ▁, ▁ Mom ▁, ▁ we ▁ are ▁ in ▁ the ▁ port ▁. ▁ It ▁ is ▁ our ▁ secret ▁ port ▁. ▁ Do ▁ you ▁ want ▁ to ▁ come ▁ in ▁?" ▁\\n ▁\\n ▁Mom ▁ says ▁, ▁ " ▁Sure ▁, ▁ I ▁ would ▁ love ▁ to ▁ come ▁ in ▁. ▁ Can ▁ I ▁ fit ▁ in ▁ the ▁ hole ▁?" ▁\\n ▁\\n ▁L ▁ila ▁ and ▁ Tom ▁ say ▁, ▁ " ▁Yes ▁, ▁ Mom ▁, ▁ you ▁ can ▁ fit ▁ in ▁ the ▁ hole ▁. ▁ Come ▁ in ▁, ▁ come ▁ in ▁." ▁\\n ▁\\n ▁Mom ▁ craw ▁ls ▁ in ▁ the ▁ hole ▁. ▁ She ▁ joins ▁ L ▁ila ▁ and ▁ Tom ▁ in ▁ the ▁ port ▁. ▁ They ▁ hug ▁ and ▁ kiss ▁. ▁ They ▁ are ▁ happy ▁. ▁ They ▁ have ▁ a ▁ secret ▁ port ▁. ▁

</details>

Example 9: `They ▁ wondered ▁ if ▁ it ▁ was ▁ a ▁ lion ▁ or ▄ a █ bear ▃ or ▅ a ▇ monster ▃. ▁`

<details>
<summary>Click to see all of example 9</summary>

> L ▁ily ▁ and ▁ Ben ▁ were ▁ playing ▁ in ▁ the ▁ park ▁ with ▁ their ▁ toy ▁ cars ▁. ▁ They ▁ liked ▁ to ▁ make ▁ noises ▁ as ▁ they ▁ zoom ▁ed ▁ them ▁ around ▁ the ▁ grass ▁ and ▁ the ▂ sand ▁. ▁ Lily ▁ had ▁ a ▁ red ▁ car ▁ and ▁ Ben ▁ had ▁ a ▁ blue ▁ car ▁. ▁ They ▁ pretended ▁ they ▁ were ▁ racing ▁ each ▁ other ▁. ▁\\n ▁\\n ▁Suddenly ▁, ▁ they ▁ heard ▁ a ▁ loud ▁ roar ▁ from ▁ behind ▁ the ▁ trees ▁. ▁ They ▁ stopped ▁ playing ▁ and ▁ looked ▁ at ▁ each ▁ other ▁. ▁ They ▁ felt ▁ scared ▁. ▁ What ▁ could ▁ make ▁ such ▁ a ▁ big ▁ noise ▁? ▁ They ▁ wondered ▁ if ▁ it ▁ was ▁ a ▁ lion ▁ or ▄ a █ bear ▃ or ▅ a ▇ monster ▃. ▁\\n ▁\\n ▁They ▁ saw ▁ a ▁ flash ▁ of ▁ light ▁ from ▁ behind ▁ the ▁ trees ▁. ▁ It ▁ was ▁ bright ▁ and ▁ yellow ▁. ▁ They ▁ heard ▁ another ▁ roar ▁, ▂ louder ▁ than ▁ before ▁. ▁ They ▁ saw ▁ something ▁ big ▁ and ▁ brown ▁ come ▁ out ▁ of ▁ the ▁ trees ▁. ▁ It ▁ had ▁ a ▁ wide ▁ mouth ▁ and ▁ sharp ▁ teeth ▁. ▁ It ▁ was ▁ a ▁ crocod ▁ile ▁! ▁\\n ▁\\n ▁L ▁ily ▁ and ▁ Ben ▁ dropped ▁ their ▁ toy ▁ cars ▁ and ▁ ran ▁ away ▁ as ▁ fast ▁ as ▁ they ▁ could ▁. ▁ They ▁ screamed ▁ for ▁ help ▁. ▁ They ▁ saw ▁ their ▁ mom ▁ and ▁ dad ▁ sitting ▁ on ▁ a ▁ bench ▁ nearby ▁. ▁ They ▁ ran ▁ to ▁ them ▁ and ▁ hugged ▁ them ▁. ▁ They ▁ told ▁ them ▁ what ▁ they ▁ saw ▁. ▁\\n ▁\\n ▁Mom ▁ and ▁ dad ▁ looked ▁ at ▁ the ▁ trees ▁. ▁ They ▁ saw ▁ the ▁ light ▁ and ▁ the ▁ crocod ▁ile ▁ too ▁. ▁ But ▁ they ▁ also ▁ saw ▁ something ▁ else ▁. ▁ They ▁ saw ▁ a ▁ man ▁ with ▁ a ▁ camera ▁ and ▁ a ▂ speaker ▁. ▁ He ▁ was ▁ making ▁ the ▁ roar ▁ and ▁ the ▁ light ▁. ▁ He ▁ was ▁ making ▁ a ▁ movie ▁ about ▁ crocod ▁iles ▁. ▁\\n ▁\\n ▁Mom ▁ and ▁ dad ▁ laughed ▁ and ▁ explained ▁ to ▁ Lily ▁ and ▁ Ben ▁ that ▁ it ▁ was ▁ not ▁ a ▁ real ▁ crocod ▁ile ▁. ▁ It ▁ was ▁ a ▁ fake ▁ one ▁. ▁ It ▁ was ▁ just ▁ for ▁ fun ▁. ▁ They ▁ said ▁ they ▁ were ▁ sorry ▁ for ▁ sc ▁aring ▁ them ▁. ▁ They ▁ said ▁ they ▁ could ▁ go ▁ and ▁ see ▁ the ▁ movie ▁ when ▁ it ▁ was ▁ done ▁. ▁\\n ▁\\n ▁L ▁ily ▁ and ▁ Ben ▁ felt ▁ better ▁. ▁ They ▁ were ▁ not ▁ scared ▁ anymore ▁. ▁ They ▁ were ▁ curious ▁. ▁ They ▁ wanted ▁ to ▁ see ▁ the ▁ movie ▁ too ▁. ▁ They ▁ picked ▁ up ▁ their ▁ toy ▁ cars ▁ and ▁ went ▁ to ▁ the ▁ man ▁. ▁ They ▁ said ▁ hello ▁ and ▁ asked ▁ him ▁ questions ▁. ▁ He ▁ was ▁ nice ▁ and ▁ showed ▁ them ▁ how ▁ he ▁ made ▁ the ▁ roar ▁ and ▁ the ▁ light ▁. ▁ He ▁ let ▁ them ▁ touch ▁ the ▁ fake ▁ crocod ▁ile ▁. ▁ It ▁ was ▁ soft ▁ and ▁ smooth ▁. ▁\\n ▁\\n ▁L ▁ily ▁ and ▁ Ben ▁ had ▁ fun ▁. ▁ They ▁ learned ▁ something ▁ new ▁. ▁ They ▁ liked ▁ the ▁ movie ▁ man ▁ and ▁ the ▁ fake ▁ crocod ▁ile ▁. ▁ They ▁ said ▁ thank ▁ you ▁ and ▁ goodbye ▁. ▁ They ▁ went ▁ back ▁ to ▁ playing ▁ with ▁ their ▁ toy ▁ cars ▁. ▁ They ▁ still ▁ made ▁ noises ▁, ▁ but ▁ they ▁ also ▁ made ▁ ro ▁ars ▁ and ▁ lights ▁. ▁ They ▁ pretended ▁ they ▁ were ▁ crocod ▁iles ▁ too ▁. ▁

</details>

Example 10: `Something ▁ that ▁ likes ▁ to ▁ fly ▁. ▁ Like ▁ a ▂ bird ▂ or ▁ a ▆ plane ▄ or ▄ a █ k ▅ite ▄,\" ▁ Lily ▁ said ▁`

<details>
<summary>Click to see all of example 10</summary>

> L ▁ily ▁ and ▁ Ben ▁ were ▁ playing ▁ with ▁ their ▁ toys ▁ in ▁ the ▁ backyard ▁. ▁ They ▁ had ▁ a ▁ lot ▁ of ▁ fun ▁ cars ▁, ▁ trucks ▁, ▁ and ▁ trains ▁ to ▁ make ▁ noises ▁ and ▁ zoom ▁ around ▁. ▁ But ▁ Lily ▁'s ▁ favorite ▁ toy ▁ was ▁ her ▁ crane ▁. ▁ It ▁ was ▁ big ▁ and ▁ yellow ▁ and ▁ had ▁ a ▁ long ▁ arm ▁ that ▁ could ▁ lift ▁ things ▁ up ▁ and ▁ down ▁. ▁\\n ▁\\n ▁" ▁Look ▁, ▁ Ben ▁, ▁ I ▁ can ▁ put ▁ this ▁ ball ▁ on ▁ the ▁ crane ▁ and ▁ make ▁ it ▁ fly ▁!" ▁ Lily ▁ said ▁, ▁ as ▁ she ▁ hooked ▁ the ▁ ball ▁ to ▁ the ▁ crane ▁'s ▁ arm ▁ and ▁ pressed ▁ a ▁ button ▁. ▁ The ▁ crane ▁ lifted ▁ the ▁ ball ▁ high ▁ in ▁ the ▁ air ▁ and ▁ then ▁ lowered ▁ it ▁ down ▁ again ▁. ▁\\n ▁\\n ▁" ▁Wow ▁, ▁ that ▁'s ▁ cool ▁, ▁ Lily ▁! ▁ Can ▁ I ▁ try ▁?" ▁ Ben ▁ asked ▁, ▁ reaching ▁ for ▁ the ▁ crane ▁. ▁\\n ▁\\n ▁" ▁OK ▁, ▁ but ▁ be ▁ careful ▁. ▁ It ▁'s ▁ my ▁ crane ▁ and ▁ I ▁ love ▁ it ▁ very ▁ much ▁," ▁ Lily ▁ said ▁, ▁ handing ▁ the ▁ crane ▁ to ▁ Ben ▁. ▁\\n ▁\\n ▁Ben ▁ smiled ▁ and ▁ took ▁ the ▁ crane ▁. ▁ He ▁ looked ▁ for ▁ something ▁ to ▁ put ▁ on ▁ the ▁ crane ▁. ▁ He ▁ saw ▁ a ▁ small ▁, ▁ fluffy ▁, ▁ white ▁ cat ▁ sleeping ▁ on ▁ a ▁ chair ▁. ▁ The ▁ cat ▁ was ▁ adorable ▁ and ▁ looked ▁ very ▁ soft ▁. ▁\\n ▁\\n ▁" ▁I ▁ know ▁, ▁ I ▁'ll ▁ put ▁ the ▁ cat ▁ on ▁ the ▁ crane ▁ and ▁ make ▁ it ▁ fly ▁ too ▁!" ▁ Ben ▁ said ▁, ▁ as ▁ he ▁ grabbed ▁ the ▁ cat ▁ and ▁ tried ▁ to ▁ hook ▁ it ▁ to ▁ the ▁ crane ▁'s ▁ arm ▁. ▁\\n ▁\\n ▁But ▁ the ▁ cat ▁ did ▁ not ▁ like ▁ that ▁ at ▁ all ▁. ▁ It ▁ his ▁sed ▁ and ▁ scratched ▁ and ▁ bit ▁ Ben ▁'s ▁ hand ▁. ▁ Ben ▁ dropped ▁ the ▁ cat ▁ and ▁ the ▁ crane ▁ and ▁ cried ▁ out ▁ loud ▁. ▁\\n ▁\\n ▁" ▁O ▁w ▁, ▁ ow ▁, ▁ ow ▁! ▁ The ▁ cat ▁ hurt ▁ me ▁!" ▁ Ben ▁ said ▁, ▁ holding ▁ his ▁ hand ▁. ▁\\n ▁\\n ▁L ▁ily ▁ ran ▁ to ▁ Ben ▁ and ▁ saw ▁ his ▁ hand ▁ bleeding ▁. ▁ She ▁ felt ▁ sorry ▁ for ▁ him ▁ and ▁ hugged ▁ him ▁. ▁\\n ▁\\n ▁" ▁I ▁'m ▁ sorry ▁, ▁ Ben ▁. ▁ The ▁ cat ▁ didn ▁'t ▁ want ▁ to ▁ fly ▁. ▁ It ▁ was ▁ scared ▁ and ▁ angry ▁. ▁ You ▁ should ▁ not ▁ put ▁ the ▁ cat ▁ on ▁ the ▁ crane ▁. ▁ That ▁'s ▁ not ▁ nice ▁," ▁ Lily ▁ said ▁. ▁\\n ▁\\n ▁" ▁I ▁'m ▁ sorry ▁, ▁ Lily ▁. ▁ I ▁ didn ▁'t ▁ know ▁ the ▁ cat ▁ would ▁ do ▁ that ▁. ▁ I ▁ just ▁ wanted ▁ to ▁ have ▁ fun ▁. ▁ I ▁'m ▁ sorry ▁ I ▁ broke ▁ your ▁ crane ▁ too ▁," ▁ Ben ▁ said ▁. ▁\\n ▁\\n ▁L ▁ily ▁ looked ▁ at ▁ her ▁ crane ▁ and ▁ saw ▁ that ▁ it ▁ was ▁ not ▁ broken ▁. ▁ It ▁ was ▁ just ▁ a ▁ little ▂ dirty ▁. ▁ She ▁ picked ▁ it ▁ up ▁ and ▁ wiped ▁ it ▁ with ▁ a ▁ cloth ▁. ▁\\n ▁\\n ▁" ▁It ▁'s ▁ OK ▁, ▁ Ben ▁. ▁ My ▁ crane ▁ is ▁ fine ▁. ▁ It ▁'s ▁ still ▁ big ▁ and ▁ yellow ▁ and ▁ can ▁ lift ▁ things ▁ up ▁ and ▁ down ▁. ▁ But ▁ next ▁ time ▁, ▁ let ▁'s ▁ put ▁ something ▁ else ▁ on ▁ the ▁ crane ▁. ▁ Something ▁ that ▁ likes ▁ to ▁ fly ▁. ▁ Like ▁ a ▂ bird ▂ or ▁ a ▆ plane ▄ or ▄ a █ k ▅ite ▄," ▁ Lily ▁ said ▁. ▁\\n ▁\\n ▁" ▁OK ▁, ▁ Lily ▁. ▁ That ▁ sounds ▁ like ▁ a ▁ good ▁ idea ▁. ▁ I ▁'m ▁ sorry ▁ I ▁ made ▁ you ▁ and ▁ the ▁ cat ▁ sad ▁. ▁ Can ▁ we ▁ still ▁ be ▁ friends ▁?" ▁ Ben ▁ asked ▁. ▁\\n ▁\\n ▁" ▁Of ▁ course ▁, ▁ Ben ▁. ▁ We ▁ are ▁ always ▁ friends ▁. ▁ And ▁ the ▁ cat ▁ is ▁ OK ▁ too ▁. ▁ Look ▁, ▁ it ▁'s ▁ pur ▁ring ▁ on ▁ the ▁ chair ▁ again ▁. ▁ It ▁'s ▁ adorable ▁," ▁ Lily ▁ said ▁. ▁\\n ▁\\n ▁They ▁ smiled ▁ and ▁ hugged ▁ and ▁ went ▁ back ▁ to ▁ playing ▁ with ▁ their ▁ toys ▁. ▁ They ▁ had ▁ a ▁ lot ▁ of ▁ fun ▁ with ▁ the ▁ crane ▁ and ▁ other ▁ things ▁ that ▁ liked ▁ to ▁ fly ▁. ▁ And ▁ they ▁ never ▁ put ▁ the ▁ cat ▁ on ▁ the ▁ crane ▁ again ▁. ▁

</details>

### Feature 2

My interpretation: This feature activates on "time" in "it was time to/for",
usually in a phrase like "the characters were having fun, but then it was time for something less fun".

GPT-4o's interpretation: "'time' is always strongly highlighted."

Example 1: `It ▁ was ▁ so ▁ much ▁ fun ▁! ▁  ▁\n ▁\n ▁But ▁ then ▁ it ▁ was ▁ time █ to ▁ go ▁ home ▁. ▁`

<details>
<summary>Click to see all of example 1</summary>

> Today ▁ was ▁ ordinary ▁. ▁ Emma ▁â ▁€ ▁™ ▁s ▁ mom ▁ took ▁ her ▁ to ▁ the ▁ park ▁. ▁ They ▁ played ▁ tag ▁, ▁ and ▁ swung ▁ on ▁ the ▁ swings ▁. ▁ It ▁ was ▁ so ▁ much ▁ fun ▁! ▁  ▁\\n ▁\\n ▁But ▁ then ▁ it ▁ was ▁ time █ to ▁ go ▁ home ▁. ▁ Emma ▁ felt ▁ sad ▁. ▁ She ▁ hugged ▁ her ▁ mom ▁ and ▁ said ▁, ▁ � ▁� ▁€ ▁� ▁� ▁I ▁ miss ▁ the ▁ park ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁Her ▁ mom ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� ▁€ ▁� ▁� ▁Me ▁ too ▁. ▁ But ▁ it ▁ will ▁ still ▁ be ▁ here ▁ tomorrow ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁So ▁, ▁ Emma ▁ said ▁ goodbye ▁ and ▁ went ▁ inside ▁. ▁ She ▁ wished ▁ she ▁ could ▁ stay ▁ and ▁ play ▁ forever ▁, ▁ but ▁ tomorrow ▁ would ▁ come ▁. ▁ And ▁ she ▁ knew ▁ she ▁ would ▁ miss ▁ the ▁ park ▁ today ▁, ▁ but ▁ she ▁ could ▁ come ▁ back ▁ tomorrow ▁ to ▁ play ▁ again ▁. ▁

</details>

Example 2: `They ▁ had ▁ so ▁ much ▁ fun ▁ pretending ▁ to ▁ be ▁ a ▁ plane ▁, ▁ but ▁ it ▁ was ▁ time █ for ▁ them ▁ to ▁ go ▁ home ▁. ▁`

<details>
<summary>Click to see all of example 2</summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ were ▁ two ▁ friends ▁, ▁ Jack ▁ and ▁ Jill ▁. ▁ They ▁ were ▁ both ▁ so ▁ excited ▁ to ▁ play ▁ together ▁ outdoors ▁. ▁ They ▁ looked ▁ up ▁ and ▁ saw ▁ a ▁ beautiful ▁ white ▁ bird ▁ soaring ▁ in ▁ the ▁ sky ▁. ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁Wow ▁, ▁ look ▁ at ▁ that ▁ bird ▁ soar ▁! ▁â ▁€ ▁ shouted ▁ Jack ▁. ▁  ▁\\n ▁\\n ▁J ▁ill ▁ replied ▁, ▁ � ▁� ▁€ ▁� ▁� ▁Let ▁â ▁€ ▁™ ▁s ▁ be ▁ like ▁ the ▁ bird ▁ and ▁ soar ▁ too ▁! ▁â ▁€ ▁\\n ▁\\n ▁Jack ▁ saw ▁ a ▁ broken ▁ branch ▁ on ▁ the ▁ ground ▁, ▁ so ▁ he ▁ picked ▁ it ▁ up ▁ and ▁ said ▁, ▁ � ▁� ▁€ ▁� ▁� ▁Let ▁â ▁€ ▁™ ▁s ▁ pretend ▁ this ▁ branch ▁ is ▁ like ▁ a ▁ plane ▁. ▁â ▁€ ▁\\n ▁\\n ▁So ▁ Jack ▁ and ▁ Jill ▁ hugged ▁ the ▁ branch ▁, ▁ closed ▁ their ▁ eyes ▁ and ▁ imagined ▁ they ▁ were ▁ soaring ▁ in ▁ the ▁ sky ▁. ▁ They ▁ hum ▁med ▁ and ▁ soared ▁ around ▁ each ▁ other ▁, ▁ pretending ▁ to ▁ make ▁ plane ▁ noises ▁. ▁\\n ▁\\n ▁They ▁ had ▁ so ▁ much ▁ fun ▁ pretending ▁ to ▁ be ▁ a ▁ plane ▁, ▁ but ▁ it ▁ was ▁ time █ for ▁ them ▁ to ▁ go ▁ home ▁. ▁ They ▁ hugged ▁ the ▁ branch ▁ one ▁ last ▁ time ▁, ▁ thanked ▁ it ▁ for ▁ the ▁ fun ▁, ▁ and ▁ said ▁ goodbye ▁. ▁

</details>

Example 3: `He ▁ felt ▁ so ▁ happy ▁ for ▁ his ▁ new ▁ adventure ▁. ▁\n ▁\n ▁The ▁ sun ▁ was ▁ setting ▁ and ▁ it ▁ was ▁ time █ for ▁ Ted ▁ to ▁ go ▁ home ▁, ▁`

<details>
<summary>Click to see all of example 3</summary>

> Once ▁ there ▁ was ▁ a ▁ nice ▁ little ▁ boy ▁ named ▁ Ted ▁. ▁ He ▁ liked ▁ to ▁ climb ▁ lots ▁ of ▁ things ▁ like ▁ fences ▁, ▁ trees ▁ and ▁ rocks ▁. ▁ Today ▁ Ted ▁ woke ▁ up ▁ feeling ▁ extra ▁ special ▁. ▁ He ▁ decided ▁ to ▁ explore ▁ a ▁ new ▁ place ▁ and ▁ try ▁ something ▁ he ▁ had ▁ never ▁ done ▁ before ▁. ▁\\n ▁\\n ▁Ted ▁ thought ▁ that ▁ climbing ▁ a ▁ mountain ▁ would ▁ be ▁ exciting ▁. ▁ So ▁ he ▁ asked ▁ his ▁ mom ▁, ▁ � ▁� ▁€ ▁� ▁� ▁Mom ▁ can ▁ I ▁ go ▁ mountain ▁ climbing ▁? ▁â ▁€ ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁Oh ▁ my ▁, ▁â ▁€ ▁ said ▁ his ▁ mom ▁, ▁ � ▁� ▁€ ▁� ▁� ▁that ▁'s ▁ a ▁ little ▁ too ▁ difficult ▁ for ▁ you ▁ right ▁ now ▁ Ted ▁, ▁ but ▁ I ▁ know ▁ a ▁ place ▁ you ▁ can ▁ go ▁. ▁ It ▁ has ▁ a ▁ kind ▁ of ▁ flame ▁ at ▁ the ▁ top ▁. ▁ It ▁'s ▁ called ▁ a ▁ lighthouse ▁. ▁â ▁€ ▁\\n ▁\\n ▁Ted ▁'s ▁ face ▁ lit ▁ up ▁ with ▁ excitement ▁. ▁ He ▁ knew ▁ his ▁ mom ▁ was ▁ right ▁, ▁ climbing ▁ a ▁ lighthouse ▁ would ▁ be ▁ much ▁ better ▁ than ▁ climbing ▁ a ▁ mountain ▁. ▁\\n ▁\\n ▁Ted ▁ put ▁ his ▁ best ▁ shoes ▁ and ▁ grabbed ▁ his ▁ hat ▁. ▁ He ▁ said ▁ goodbye ▁ to ▁ his ▁ mom ▁ and ▁ started ▁ running ▁ towards ▁ the ▁ lighthouse ▁. ▁\\n ▁\\n ▁As ▁ Ted ▁ was ▁ getting ▁ closer ▁ he ▁ could ▁ see ▁ the ▁ huge ▁ tower ▁ and ▁ the ▁ kind ▁ flame ▁ at ▁ the ▁ top ▁. ▁ He ▁ stopped ▁ to ▁ take ▁ a ▁ breath ▁ and ▁ then ▁ climbed ▁ the ▁ stairs ▁ until ▁ he ▁ reached ▁ the ▁ top ▁. ▁\\n ▁\\n ▁Ted ▁ looked ▁ around ▁, ▁ he ▁ could ▁ see ▁ for ▁ miles ▁ in ▁ every ▁ direction ▁. ▁ He ▁ felt ▁ so ▁ happy ▁ for ▁ his ▁ new ▁ adventure ▁. ▁\\n ▁\\n ▁The ▁ sun ▁ was ▁ setting ▁ and ▁ it ▁ was ▁ time █ for ▁ Ted ▁ to ▁ go ▁ home ▁, ▁ but ▁ he ▁ knew ▁ he ▁ would ▁ be ▁ back ▁ to ▁ climb ▁ his ▁ lighthouse ▁ again ▁ soon ▁. ▁

</details>

Example 4: `Tom ▁ and ▁ Sue ▁ laughed ▁ and ▁ had ▁ a ▁ lot ▁ of ▁ fun ▁ together ▁. ▁  ▁\n ▁\n ▁Finally ▁, ▁ it ▁ was ▁ time █ for ▁ Tom ▁ to ▁ go ▁ home ▁. ▁`

<details>
<summary>Click to see all of example 4</summary>

> Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ deaf ▁ boy ▁ called ▁ Tom ▁. ▁ Tom ▁ liked ▁ to ▁ go ▁ for ▁ walks ▁. ▁ One ▁ day ▁, ▁ Tom ▁ asked ▁ his ▁ Mom ▁my ▁ if ▁ he ▁ could ▁ go ▁ for ▁ a ▁ visit ▁. ▁ His ▁ Mom ▁my ▁ said ▁ � ▁� ▁€ ▁� ▁� ▁Yes ▁, ▁ you ▁ can ▁ go ▁ for ▁ a ▁ visit ▁. ▁ But ▁ make ▁ sure ▁ you ▁â ▁€ ▁™ ▁re ▁ home ▁ before ▁ dinner ▁. ▁â ▁€ ▁ So ▁ Tom ▁ put ▁ on ▁ his ▁ shoes ▁ to ▁ go ▁ for ▁ a ▁ walk ▁. ▁\\n ▁\\n ▁On ▁ his ▁ walk ▁, ▁ Tom ▁ met ▁ a ▁ deaf ▁ girl ▁ called ▁ Sue ▁. ▁ Sue ▁ said ▁ to ▁ Tom ▁, ▁ � ▁� ▁€ ▁� ▁� ▁Hi ▁! ▁ What ▁ are ▁ you ▁ doing ▁? ▁â ▁€ ▁ Tom ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� ▁€ ▁� ▁� ▁I ▁â ▁€ ▁™ ▁m ▁ going ▁ for ▁ a ▁ visit ▁. ▁ Would ▁ you ▁ like ▁ to ▁ come ▁ too ▁? ▁â ▁€ ▁ Sue ▁ nodded ▁, ▁ so ▁ Tom ▁ and ▁ Sue ▁ went ▁ for ▁ a ▁ walk ▁ together ▁. ▁  ▁\\n ▁\\n ▁They ▁ went ▁ to ▁ a ▁ park ▁ and ▁ looked ▁ at ▁ the ▁ birds ▁. ▁ They ▁ watched ▁ the ▁ squirrel ▁s ▁ and ▁ played ▁ on ▁ the ▁ swings ▁. ▁ Tom ▁ and ▁ Sue ▁ laughed ▁ and ▁ had ▁ a ▁ lot ▁ of ▁ fun ▁ together ▁. ▁  ▁\\n ▁\\n ▁Finally ▁, ▁ it ▁ was ▁ time █ for ▁ Tom ▁ to ▁ go ▁ home ▁. ▁ He ▁ said ▁ goodbye ▁ to ▁ Sue ▁ and ▁ thanked ▁ her ▁ for ▁ coming ▁ with ▁ him ▁ on ▁ his ▁ visit ▁. ▁ Then ▁ Tom ▁ waved ▁ goodbye ▁, ▁ and ▁ went ▁ for ▁ a ▁ walk ▁ back ▁ home ▁. ▁

</details>

Example 5: `it ▁ was ▁ so ▁ much ▁ fun ▁. ▁  ▁\n ▁\n ▁But ▁ then ▁ one ▁ day ▁ Johnny ▁'s ▁ mom ▁ said ▁ it ▁ was ▁ time █ for ▁ bed ▁. ▁`

<details>
<summary>Click to see all of example 5</summary>

> Once ▁ upon ▁ a ▁ time ▁ there ▁ was ▁ a ▁ boy ▁ called ▁ Johnny ▁. ▁ He ▁ was ▁ very ▁ excited ▁ one ▁ day ▁ because ▁ he ▁ was ▁ going ▁ to ▁ learn ▁ a ▁ new ▁ thing ▁ � ▁� ▁€ ▁� ▁� ▁ how ▁ to ▁ sign ▁. ▁ His ▁ teacher ▁ taught ▁ him ▁ how ▁ to ▁ make ▁ lots ▁ of ▁ different ▁ signs ▁ with ▁ his ▁ hands ▁ and ▁ they ▁ were ▁ very ▁ fun ▁. ▁  ▁\\n ▁\\n ▁Next ▁, ▁ his ▁ mom ▁ gave ▁ him ▁ a ▁ ball ▁ of ▁ wool ▁. ▁ It ▁ was ▁ soft ▁ and ▁ fluffy ▁ and ▁ he ▁ really ▁ liked ▁ it ▁. ▁ He ▁ played ▁ with ▁ it ▁ every ▁ day ▁. ▁ He ▁ made ▁ lots ▁ of ▁ shapes ▁ and ▁ patterns ▁ with ▁ it ▁, ▁ like ▁ a ▁ hat ▁ and ▁ a ▁ scarf ▁. ▁ His ▁ mom ▁ cl ▁apped ▁ and ▁ smiled ▁, ▁ it ▁ was ▁ so ▁ much ▁ fun ▁. ▁  ▁\\n ▁\\n ▁But ▁ then ▁ one ▁ day ▁ Johnny ▁'s ▁ mom ▁ said ▁ it ▁ was ▁ time █ for ▁ bed ▁. ▁ But ▁ bed ▁time ▁ was ▁ so ▁ boring ▁ � ▁� ▁€ ▁� ▁� ▁ Johnny ▁ wanted ▁ to ▁ keep ▁ playing ▁ with ▁ the ▁ wool ▁ instead ▁! ▁ He ▁ tried ▁ to ▁ sneak ▁ it ▁ in ▁ his ▁ bedroom ▁ but ▁ his ▁ mom ▁ caught ▁ him ▁ and ▁ said ▁ he ▁ had ▁ to ▁ go ▁ to ▁ sleep ▁. ▁  ▁\\n ▁\\n ▁Johnny ▁ sighed ▁, ▁ but ▁ then ▁ he ▁ thought ▁ of ▁ something ▁. ▁ He ▁ made ▁ a ▁ sign ▁ with ▁ his ▁ hands ▁ and ▁ his ▁ mom ▁ smiled ▁ � ▁� ▁€ ▁� ▁� ▁ it ▁ was ▁ his ▁ own ▁ sign ▁ for ▁ � ▁� ▁€ ▁� ▁� ▁w ▁ool ▁â ▁€ ▁. ▁ His ▁ mom ▁ gave ▁ him ▁ the ▁ wool ▁ and ▁ he ▁ sn ▁ugg ▁led ▁ with ▁ it ▁ all ▁ the ▁ way ▁ to ▁ bed ▁. ▁\\n ▁\\n ▁The ▁ End ▁. ▁

</details>

Example 6: `They ▁ waited ▁ for ▁ a ▁ long ▁ time ▁, ▁ but ▁ the ▁ raven ▁ never ▁ came ▁ back ▁. ▁ Eventually ▁, ▁ it ▁ was ▁ time █ to ▁ go ▁ home ▁`

<details>
<summary>Click to see all of example 6</summary>

> James ▁ and ▁ his ▁ mom ▁ were ▁ walking ▁ to ▁ the ▁ park ▁ one ▁ day ▁. ▁ James ▁ saw ▁ a ▁ beautiful ▁ raven ▁ flying ▁ in ▁ the ▁ sky ▁. ▁ He ▁ asked ▁ his ▁ mom ▁, ▁ � ▁� ▁€ ▁˜ ▁what ▁ is ▁ that ▁? ▁â ▁€ ▁™ ▁ She ▁ said ▁ it ▁ was ▁ a ▁ raven ▁. ▁ He ▁ asked ▁ what ▁ a ▁ raven ▁ was ▁. ▁ Mom ▁ said ▁, ▁ � ▁� ▁€ ▁� ▁� ▁A ▁ raven ▁ is ▁ a ▁ big ▁, ▁ black ▁ bird ▁. ▁â ▁€ ▁\\n ▁\\n ▁James ▁ wanted ▁ to ▁ see ▁ the ▁ raven ▁, ▁ so ▁ his ▁ mom ▁ and ▁ him ▁ ran ▁ to ▁ the ▁ park ▁. ▁ But ▁, ▁ when ▁ they ▁ arrived ▁, ▁ the ▁ raven ▁ was ▁ gone ▁. ▁ James ▁ asked ▁ where ▁ the ▁ raven ▁ went ▁ and ▁ his ▁ mom ▁ said ▁, ▁ � ▁� ▁€ ▁� ▁� ▁It ▁ was ▁ an ▁ unknown ▁ bird ▁, ▁ so ▁ we ▁ won ▁â ▁€ ▁™ ▁t ▁ know ▁ where ▁ it ▁ went ▁. ▁â ▁€ ▁\\n ▁\\n ▁James ▁ and ▁ his ▁ mom ▁ sat ▁ in ▁ the ▁ park ▁, ▁ looking ▁ for ▁ the ▁ raven ▁. ▁ They ▁ waited ▁ for ▁ a ▁ long ▁ time ▁, ▁ but ▁ the ▁ raven ▁ never ▁ came ▁ back ▁. ▁ Eventually ▁, ▁ it ▁ was ▁ time █ to ▁ go ▁ home ▁ and ▁ the ▁ raven ▁ had ▁ not ▁ returned ▁. ▁ James ▁ asked ▁ what ▁ happened ▁ to ▁ the ▁ bird ▁ and ▁ his ▁ mom ▁ answered ▁, ▁ � ▁� ▁€ ▁� ▁� ▁We ▁ may ▁ never ▁ know ▁. ▁ This ▁ is ▁ the ▁ end ▁ of ▁ the ▁ raven ▁. ▁â ▁€ ▁  ▁\\n ▁\\n ▁James ▁ was ▁ sad ▁ because ▁ his ▁ mom ▁ said ▁ the ▁ raven ▁ was ▁ gone ▁. ▁ He ▁ told ▁ his ▁ mom ▁, ▁ � ▁� ▁€ ▁� ▁� ▁I ▁ wish ▁ I ▁ could ▁ have ▁ found ▁ the ▁ raven ▁. ▁â ▁€ ▁ His ▁ mom ▁ replied ▁, ▁ " ▁Me ▁ too ▁, ▁ James ▁. ▁ Me ▁ too ▁." ▁

</details>

Example 7: `So ▁ the ▁ family ▁ settled ▁ at ▁ the ▁ playground ▁ and ▁ had ▁ a ▁ wonderful ▁ time ▁ playing ▁ and ▁ exploring ▁. ▁ When ▁ it ▁ was ▁ time █ to ▁ go ▁, ▁`

<details>
<summary>Click to see all of example 7</summary>

> Once ▁ upon ▁ a ▁ time ▁ there ▁ was ▁ a ▁ family ▁ that ▁ was ▁ very ▁ careful ▁. ▁ On ▁ an ▁ especially ▁ careful ▁ day ▁, ▁ they ▁ decided ▁ to ▁ go ▁ on ▁ an ▁ adventure ▁. ▁ The ▁ family ▁ loaded ▁ up ▁ their ▁ luggage ▁ into ▁ the ▁ car ▁ and ▁ everyone ▁ settled ▁ in ▁ for ▁ the ▁ ride ▁. ▁  ▁\\n ▁\\n ▁When ▁ they ▁ reached ▁ their ▁ destination ▁, ▁ the ▁ family ▁ unloaded ▁ their ▁ luggage ▁ and ▁ set ▁ off ▁ to ▁ explore ▁. ▁ As ▁ they ▁ were ▁ walking ▁, ▁ the ▁ 3 ▁ year ▁ old ▁ noticed ▁ something ▁ in ▁ the ▁ distance ▁. ▁  ▁\\n ▁\\n ▁The ▁ 3 ▁ year ▁ old ▁ excited ▁ly ▁ said ▁ to ▁ the ▁ parents ▁, ▁ " ▁What ▁ is ▁ that ▁? ▁â ▁€ ▁\\n ▁\\n ▁The ▁ parents ▁ replied ▁, ▁ � ▁� ▁€ ▁� ▁� ▁That ▁ looks ▁ like ▁ a ▁ playground ▁! ▁ Let ▁'s ▁ go ▁ check ▁ it ▁ out ▁! ▁ Be ▁ careful ▁ though ▁ and ▁ make ▁ sure ▁ your ▁ luggage ▁ is ▁ secure ▁. ▁â ▁€ ▁\\n ▁\\n ▁So ▁ the ▁ family ▁ settled ▁ at ▁ the ▁ playground ▁ and ▁ had ▁ a ▁ wonderful ▁ time ▁ playing ▁ and ▁ exploring ▁. ▁ When ▁ it ▁ was ▁ time █ to ▁ go ▁, ▁ the ▁ family ▁ carefully ▁ loaded ▁ their ▁ luggage ▁ back ▁ in ▁ the ▁ car ▁ and ▁ went ▁ on ▁ their ▁ way ▁. ▁  ▁\\n ▁\\n ▁The ▁ end ▁. ▁

</details>

Example 8: `Daisy ▁ and ▁ Mr ▁. ▁ Rabbit ▁ had ▁ plenty ▁ of ▁ fun ▁ exploring ▁ the ▁ garden ▁ together ▁, ▁ until ▁ the ▁ sun ▁ set ▁ and ▁ it ▁ was ▁ time █ for ▁ Daisy ▁ to ▁ go ▁ home ▁. ▁`

<details>
<summary>Click to see all of example 8</summary>

> Once ▁ upon ▁ a ▁ time ▁ there ▁ was ▁ a ▁ quiet ▁ sheep ▁ called ▁ Daisy ▁, ▁ who ▁ liked ▁ to ▁ wander ▁ about ▁ in ▁ the ▁ grass ▁y ▁ fields ▁ by ▁ her ▁ farm ▁ house ▁. ▁ One ▁ day ▁, ▁ Daisy ▁ boldly ▁ followed ▁ the ▁ sun ▁ as ▁ it ▁ started ▁ to ▁ set ▁ and ▁ she ▁ ended ▁ up ▁ at ▁ a ▁ tall ▁, ▁ leather ▁ gate ▁. ▁  ▁\\n ▁\\n ▁The ▁ gate ▁ had ▁ an ▁ old ▁, ▁ heavy ▁ pad ▁lock ▁ on ▁ it ▁ that ▁ Daisy ▁ could ▁ not ▁ open ▁. ▁ Suddenly ▁, ▁ she ▁ heard ▁ a ▁ friendly ▁ voice ▁ coming ▁ up ▁ from ▁ behind ▁ her ▁, ▁ � ▁� ▁€ ▁� ▁� ▁Would ▁ you ▁ like ▁ me ▁ to ▁ show ▁ you ▁ what ▁ is ▁ behind ▁ this ▁ gate ▁? ▁â ▁€ ▁\\n ▁\\n ▁Da ▁isy ▁ was ▁ startled ▁ and ▁ turned ▁ around ▁ to ▁ see ▁ a ▁ white ▁ rabbit ▁ with ▁ long ▁ ears ▁ and ▁ a ▁ long ▁, ▁ leather ▁ tail ▁. ▁ Daisy ▁ was ▁ happy ▁ to ▁ meet ▁ the ▁ rabbit ▁, ▁ who ▁ introduced ▁ himself ▁ as ▁ Mr ▁. ▁ Rabbit ▁. ▁\\n ▁\\n ▁Mr ▁. ▁ Rabbit ▁ explained ▁, ▁ � ▁� ▁€ ▁� ▁� ▁If ▁ you ▁ follow ▁ me ▁, ▁ I ▁ will ▁ lead ▁ you ▁ to ▁ a ▁ special ▁ place ▁ through ▁ the ▁ gate ▁ and ▁ you ▁ can ▁ discover ▁ what ▁ is ▁ around ▁ the ▁ bend ▁â ▁€ ▁. ▁  ▁\\n ▁\\n ▁Da ▁isy ▁ followed ▁ the ▁ rabbit ▁ beyond ▁ the ▁ leather ▁ gate ▁, ▁ not ▁ knowing ▁ what ▁ was ▁ to ▁ come ▁. ▁ The ▁ path ▁ was ▁ quiet ▁ as ▁ the ▁ two ▁ friends ▁ walked ▁ along ▁, ▁ until ▁ they ▁ came ▁ upon ▁ a ▁ wond ▁rous ▁ field ▁. ▁  ▁\\n ▁\\n ▁Del ▁ight ▁ed ▁, ▁ Daisy ▁ shouted ▁ for ▁ joy ▁, ▁ for ▁ she ▁ had ▁ discovered ▁ the ▁ most ▁ beautiful ▁ garden ▁ she ▁ had ▁ ever ▁ seen ▁. ▁ Daisy ▁ and ▁ Mr ▁. ▁ Rabbit ▁ had ▁ plenty ▁ of ▁ fun ▁ exploring ▁ the ▁ garden ▁ together ▁, ▁ until ▁ the ▁ sun ▁ set ▁ and ▁ it ▁ was ▁ time █ for ▁ Daisy ▁ to ▁ go ▁ home ▁. ▁  ▁\\n ▁\\n ▁Da ▁isy ▁ thanked ▁ Mr ▁. ▁ Rabbit ▁ for ▁ leading ▁ her ▁ to ▁ this ▁ special ▁ place ▁ and ▁ waved ▁ goodbye ▁ to ▁ him ▁ as ▁ she ▁ hopped ▁ away ▁, ▁ content ▁

</details>

Example 9: `She ▁ ran ▁ around ▁ and ▁ had ▁ lots ▁ of ▁ fun ▁. ▁ When ▁ it ▁ was ▁ time █ to ▁ go ▁ back ▁ home ▁, ▁`

<details>
<summary>Click to see all of example 9</summary>

> Once ▁ there ▁ was ▁ a ▁ brave ▁ child ▁ named ▁ Tina ▁. ▁ She ▁ wanted ▁ to ▁ go ▁ on ▁ an ▁ adventure ▁ to ▁ visit ▁ a ▁ special ▁ square ▁. ▁ So ▁ she ▁ put ▁ on ▁ her ▁ red ▁ coat ▁ and ▁ went ▁ out ▁ the ▁ door ▁. ▁  ▁\\n ▁\\n ▁On ▁ the ▁ way ▁, ▁ she ▁ met ▁ a ▁ friendly ▁ bird ▁. ▁ The ▁ bird ▁ ch ▁ir ▁ped ▁, ▁ � ▁� ▁€ ▁� ▁� ▁Where ▁ are ▁ you ▁ going ▁? ▁â ▁€ ▁  ▁\\n ▁\\n ▁â ▁€ ▁� ▁� ▁I ▁ am ▁ going ▁ to ▁ visit ▁ a ▁ special ▁ square ▁, ▁â ▁€ ▁ said ▁ Tina ▁ brave ▁ly ▁. ▁  ▁\\n ▁\\n ▁The ▁ bird ▁ said ▁, ▁ � ▁� ▁€ ▁� ▁� ▁I ▁ can ▁ show ▁ you ▁ the ▁ way ▁. ▁ Follow ▁ me ▁! ▁â ▁€ ▁ So ▁ Tina ▁ followed ▁ the ▁ bird ▁ to ▁ the ▁ square ▁. ▁  ▁\\n ▁\\n ▁When ▁ she ▁ arrived ▁, ▁ Tina ▁ saw ▁ a ▁ big ▁ park ▁ and ▁ a ▁ playground ▁. ▁ Her ▁ eyes ▁ lit ▁ up ▁ with ▁ joy ▁. ▁ She ▁ ran ▁ around ▁ and ▁ had ▁ lots ▁ of ▁ fun ▁. ▁ When ▁ it ▁ was ▁ time █ to ▁ go ▁ back ▁ home ▁, ▁ Tina ▁ thanked ▁ the ▁ bird ▁ for ▁ showing ▁ her ▁ the ▁ way ▁. ▁  ▁\\n ▁\\n ▁The ▁ bird ▁ waved ▁ goodbye ▁ and ▁ said ▁, ▁ � ▁� ▁€ ▁� ▁� ▁Good ▁bye ▁, ▁ brave ▁ Tina ▁. ▁ I ▁ hope ▁ you ▁ will ▁ visit ▁ us ▁ again ▁! ▁â ▁€ ▁\\n ▁\\n ▁T ▁ina ▁ smiled ▁ and ▁ said ▁, ▁ � ▁� ▁€ ▁� ▁� ▁Yes ▁, ▁ I ▁ will ▁! ▁â ▁€ ▁ Then ▁ she ▁ waved ▁ goodbye ▁ and ▁ went ▁ back ▁ home ▁. ▁

</details>

Example 10: `It ▁ was ▁ so ▁ much ▁ fun ▁! ▁ They ▁ even ▁ saw ▁ some ▁ ducks ▁ swimming ▁ around ▁ them ▁. ▁\n ▁\n ▁After ▁ a ▁ while ▁, ▁ the ▁ brothers ▁ decided ▁ it ▁ was ▁ time █ to ▁ go ▁ back ▁ home ▁. ▁`

<details>
<summary>Click to see all of example 10</summary>

> Once ▁ upon ▁ a ▁ time ▁ there ▁ were ▁ two ▁ brothers ▁, ▁ Harry ▁ and ▁ Rory ▁. ▁ One ▁ day ▁, ▁ the ▁ two ▁ brothers ▁ were ▁ walking ▁ by ▁ the ▁ lake ▁ when ▁ they ▁ saw ▁ something ▁ floating ▁ on ▁ the ▁ lake ▁ - ▁ it ▁ was ▁ a ▁ raft ▁. ▁\\n ▁\\n ▁Harry ▁ asked ▁ Rory ▁, ▁ " ▁What ▁ is ▁ that ▁?" ▁\\n ▁\\n ▁R ▁ory ▁ replied ▁, ▁ " ▁I ▁ think ▁ it ▁ is ▁ a ▁ raft ▁." ▁\\n ▁\\n ▁Harry ▁ was ▁ excited ▁ and ▁ said ▁, ▁ " ▁Let ▁'s ▁ go ▁ see ▁!" ▁\\n ▁\\n ▁So ▁ the ▁ two ▁ brothers ▁ went ▁ over ▁ to ▁ the ▁ raft ▁. ▁ When ▁ they ▁ got ▁ close ▁, ▁ they ▁ saw ▁ a ▁ big ▁ smile ▁y ▁ face ▁ painted ▁ on ▁ the ▁ raft ▁. ▁ They ▁ knew ▁ they ▁ were ▁ lucky ▁! ▁\\n ▁\\n ▁The ▁ two ▁ brothers ▁ climbed ▁ onto ▁ the ▁ raft ▁ and ▁ began ▁ to ▁ paddle ▁ around ▁ the ▁ lake ▁. ▁ It ▁ was ▁ so ▁ much ▁ fun ▁! ▁ They ▁ even ▁ saw ▁ some ▁ ducks ▁ swimming ▁ around ▁ them ▁. ▁\\n ▁\\n ▁After ▁ a ▁ while ▁, ▁ the ▁ brothers ▁ decided ▁ it ▁ was ▁ time █ to ▁ go ▁ back ▁ home ▁. ▁ But ▁ before ▁ they ▁ could ▁ get ▁ off ▁ the ▁ raft ▁, ▁ Harry ▁ said ▁, ▁ " ▁Wait ▁, ▁ I ▁ forgot ▁ something ▁!" ▁\\n ▁\\n ▁R ▁ory ▁ asked ▁, ▁ " ▁What ▁ did ▁ you ▁ forget ▁?" ▁\\n ▁\\n ▁Harry ▁ said ▁, ▁ " ▁I ▁ found ▁ something ▁ on ▁ the ▁ raft ▁!" ▁\\n ▁\\n ▁R ▁ory ▁ looked ▁ around ▁ and ▁ could ▁ not ▁ find ▁ anything ▁. ▁ But ▁ then ▁ Harry ▁ pulled ▁ out ▁ a ▁ big ▁ red ▁ ball ▁ from ▁ under ▁ the ▁ edge ▁ of ▁ the ▁ raft ▁. ▁ It ▁ was ▁ a ▁ lucky ▁ find ▁! ▁\\n ▁\\n ▁The ▁ two ▁ brothers ▁ waved ▁ goodbye ▁ to ▁ the ▁ raft ▁ and ▁ took ▁ the ▁ ball ▁ home ▁. ▁ They ▁ had ▁ a ▁ lot ▁ of ▁ fun ▁ and ▁ they ▁ were ▁ so ▁ lucky ▁ to ▁ find ▁ the ▁ ball ▁! ▁

</details>

## Dead features

On 1000 examples from the validation set,
each of the autoencoder's 10000 features activated at least once,
i.e. there were no dead features.

My understanding of the
[literature](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#scaling-sae-experiments)
is that dead features
are more of a problem for larger autoencoders:

> At the end of training, we defined “dead” features as those which were not active over a sample of \\(10^{7}\\) tokens. The proportion of dead features was roughly 2% for the 1M SAE, 35% for the 4M SAE, and 65% for the 34M SAE.
