# PicoGPT, orgignal code by [jaymody](https://github.com/jaymody/picoGPT)

You've seen [openai/gpt-2](https://github.com/openai/gpt-2).

You've seen [karpathy/minGPT](https://github.com/karpathy/mingpt).

You've even seen [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)!

But have you seen [picoGPT](https://github.com/jaymody/picoGPT)??!?

`picoGPT` is an unnecessarily tiny and minimal implementation of [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in plain [NumPy](https://numpy.org). The entire forward pass code is [40 lines of code](https://github.com/jaymody/picoGPT/blob/main/gpt2_pico.py#L3-L41). I wrote a related [blog post](https://jaykmody.com/blog/gpt-from-scratch/) for picoGPT.

picoGPT features:
* Fast? ❌ Nah, picoGPT is megaSLOW 🐌
* Training code? ❌ Error, 4️⃣0️⃣4️⃣ not found
* Batch inference? Sort of ✅
* top-p sampling? Sort of ✅
* Readable? `gpt2.py` ✅
* Smol??? ✅✅✅✅✅✅ YESS!!! TEENIE TINY in fact 🤏

A quick breakdown of each of the files:

* `encoder.py` contains the code for OpenAI's BPE Tokenizer, taken straight from their [gpt-2 repo](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
* `utils.py` contains the code to download and load the GPT-2 model weights, tokenizer, and hyper-parameters.
* `gpt2.py` contains the actual GPT model and generation code which we can run as a python script.
* `gpt2_pico.py` is the same as `gpt2.py`, but in even fewer lines of code. Why? Because why not 😎👍.

#### Dependencies
```bash
pip install -r requirements.txt
```
Tested on `Python 3.9.10`.

#### Usage
```bash
python gpt2.py
> "Alan Turing theorized that computers would one day become
```

Which generates

```
so intelligent that they would be able to think for themselves, and he predicted that computers would one day be able to simulate human thought.
```

You can also control the model size (one of `["124M", "355M", "774M", "1558M"]`).
