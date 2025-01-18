# Introduction

This is an imitation of Anthropic's
[Scaling monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
work, except using a far smaller language model, `roneneldan/TinyStories-33M`.

Code [here](https://github.com/TheodoreEhrenborg/tiny_stories_sae)

## Demo

I trained a sparse autoencoder with 10000 features, and
GPT-4o says feature 87 relates to

> the concepts of upward motion or reaching a peak state such as 'up', 'fly', 'high', and similar directions or motions

Let's ask the TinyStories model to complete a story starting with
"There once was a cat".
I'll nudge each activation (specifically those just after layer 2) in the direction
of feature 87.

And indeed, the cat flies:

> There once was a cat named Mittens. Mittens was very fast and loved to play with his friends. One day, Mittens saw a bird fly in the sky like an orange taxi.<br><br>
> Mittens said to his friend, "I wish I could fly like a bird." The bird replied, "That's easy, just go slow and never be foolish sometimes."<br><br>
> Mittens practiced every day, getting better and better. One day, they passed each other and Timmy said, "Wow, you're so good at flying!" Mittens' friends laughed and said, "We are like angels!"<br><br>
> From that day on, Mittens was never lazy again and always remembered to practice and race with other kids. He felt happy and free while flying high in the sky.<br>
> \<|endoftext|>
