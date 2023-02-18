import numpy as np
import time

debug = False

def gelu(x):
	return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
	exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
	return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
	mean = np.mean(x, axis=-1, keepdims=True)
	variance = np.var(x, axis=-1, keepdims=True)
	x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
	return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
	return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
	# project up
	a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

	# project back down
	x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

	return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
	return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
	# qkv projection
	x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

	# split into qkv
	qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

	# split into heads
	qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

	# causal mask to hide future inputs from being attended to
	causal_mask = (1 - np.tri(x.shape[0], dtype=np.float16)) * -1e10  # [n_seq, n_seq]


	# perform attention over each head
	out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

	# merge heads
	x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

	# out projection
	x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

	return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
	# multi-head causal self attention
	x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

	# position-wise feed forward network
	x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

	return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
	# token + positional embeddings
	x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

	# forward pass through n_layer transformer blocks
	for block in blocks:
		x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

	# projection to vocab
	x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
	return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, sentence_n):
	while True:
		logits = gpt2(inputs[-40:], **params, n_head=n_head)[-1] # model forward pass

		#Sort of top-p sampling
		s = logits.argsort()[-10:][::-1]
		slice = [x for x in s if ((1/logits[s[0]])*logits[x]>0.92)]
		next_id = np.random.choice(slice)

		# Debug output to see which token network choose
		if debug:
			print()
			for index in slice:
				if next_id == index:
					print('V', index, logits[index], encoder.decode([index]))
				else:
					print(' ', index, logits[index], encoder.decode([index]))

		#next_id = np.argmax(logits[-1])  # greedy sampling

		inputs = np.append(inputs, [next_id]) # append prediction to input
		if inputs[-1]==13:
			sentence_n-=1
		if sentence_n < 0:
			break
			
		print(encoder.decode([inputs[-1]]), end='', flush=True)
	print()

if __name__ == "__main__":
	from utils import load_encoder_hparams_and_params
	
	model_size = "1558M"
	models_dir = "models"
	
	encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
	
	while True:
		prompt = input('>')
		if prompt == '':
			continue
		# encode the input string using the BPE tokenizer
		t = time.time()
		input_ids = encoder.encode(prompt)

		# make sure we are not surpassing the max sequence length of our model
		#assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

		# generate output ids
		output_ids = generate(input_ids, params, hparams["n_head"], 3) # Max number of sentences is 5

		# decode the ids back into a string
		# encoder.decode(output_ids)
		print(time.time() - t)
