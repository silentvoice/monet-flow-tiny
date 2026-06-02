# Results Notes

The visual article is the canonical write-up for this experiment:

<https://ankurbohra.com/notes/what-training-a-tiny-text-to-image-model-looks-like>

## What Worked

- The synthetic smoke tests validate the training path without external data.
- MONET latent shards load as `32 x 16 x 16` image latents paired with
  512-number text embeddings.
- The baseline flow model can lower velocity loss on real latent shards.
- A 512-example overfit run can produce coherent decoded images, which confirms
  the data path, model, sampler, and decoder are wired together correctly.

## What Did Not Fully Work

- Broader runs learned image-like texture, lighting, and scene structure before
  they learned reliable prompt binding.
- Self-Flow-lite improved the small matched velocity-loss comparison, but lower
  loss did not automatically produce prompt-coherent images.
- Stronger CFG-style guidance could make samples more dramatic without making
  them more faithful to the prompt. The default configs do not train a full
  unconditional branch because `text_dropout_prob` is `0.0`.

## Main Boundary

The useful distinction is:

```text
image-like:      the sample resembles the image distribution
prompt-coherent: the requested subject or scene appears in the right way
```

The tiny overfit run crossed the coherence boundary. The broader run became
image-like, but prompt binding remained the next bottleneck.
