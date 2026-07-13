# Appendix D — Glossary

Every AI term used in this course, in one or two plain sentences, grouped by topic. The **Ch.** column links to where the term is explained in full.

## Core ideas

| Term | Meaning | Ch. |
|------|---------|-----|
| **Artificial intelligence (AI)** | The broad field of making computers do things that seem to need intelligence. Machine learning is the part this course teaches. | [1](../../chapters/01-what-is-ai/README.md) |
| **Machine learning (ML)** | Writing programs that improve from data instead of being told every rule. | [1](../../chapters/01-what-is-ai/README.md) |
| **Deep learning** | Machine learning using neural networks with many layers. | [1](../../chapters/01-what-is-ai/README.md) |
| **Model** | The thing being trained: a function with adjustable numbers (parameters) inside. | [1](../../chapters/01-what-is-ai/README.md) |
| **Feature** | One measurable property of an input example (a pixel brightness, a house's size). | [1](../../chapters/01-what-is-ai/README.md) |
| **Label** | The correct answer attached to a training example. | [1](../../chapters/01-what-is-ai/README.md) |
| **Supervised learning** | Learning from examples that come with labels. | [1](../../chapters/01-what-is-ai/README.md) |
| **Unsupervised learning** | Learning structure from data without labels. | [1](../../chapters/01-what-is-ai/README.md) |
| **Reinforcement learning** | Learning by acting and receiving rewards instead of labeled examples. | [30](../../chapters/30-reinforcement-learning/README.md) |
| **Training** | Adjusting a model's parameters to make its predictions better on data. | [5](../../chapters/05-linear-regression/README.md) |
| **Inference** | Using an already-trained model to make predictions. | [1](../../chapters/01-what-is-ai/README.md) |
| **Hyperparameter** | A knob the *person* chooses rather than the model learning it (learning rate, epochs, layer sizes). | [3](../../chapters/03-derivatives-and-gradients/README.md) |

## The math

| Term | Meaning | Ch. |
|------|---------|-----|
| **Vector** | A list of numbers. | [2](../../chapters/02-vectors-and-matrices/README.md) |
| **Matrix** | A grid of numbers. Neural network layers are mostly matrix multiplications. | [2](../../chapters/02-vectors-and-matrices/README.md) |
| **Dot product** | Multiply two vectors element by element and sum: the basic "how aligned are these?" operation. | [2](../../chapters/02-vectors-and-matrices/README.md) |
| **Derivative / gradient** | The slope of a function; the gradient collects the slopes for every parameter and points uphill. | [3](../../chapters/03-derivatives-and-gradients/README.md) |
| **Probability distribution** | An assignment of probabilities to all possible outcomes, summing to 1. | [4](../../chapters/04-probability-basics/README.md) |
| **Expected value** | The long-run average of a random quantity. | [4](../../chapters/04-probability-basics/README.md) |

## How models learn

| Term | Meaning | Ch. |
|------|---------|-----|
| **Parameter / weight** | One of the adjustable numbers inside a model. Training means finding good values for them. | [5](../../chapters/05-linear-regression/README.md) |
| **Bias (parameter)** | The additive parameter in a neuron or linear model; shifts the output independent of the inputs. | [5](../../chapters/05-linear-regression/README.md) |
| **Loss function** | A formula scoring how wrong the model is, as one number. Training pushes it down. | [5](../../chapters/05-linear-regression/README.md) |
| **Mean squared error (MSE)** | A loss for numeric predictions: average of (prediction − truth)². | [5](../../chapters/05-linear-regression/README.md) |
| **Cross-entropy** | A loss for classifiers measuring how surprised the model is by the true answers. Low surprise = good. | [4](../../chapters/04-probability-basics/README.md) |
| **Gradient descent** | The core training algorithm: repeatedly nudge every parameter a small step against its gradient. | [5](../../chapters/05-linear-regression/README.md) |
| **Learning rate** | The step size of gradient descent. Too big diverges, too small crawls. | [5](../../chapters/05-linear-regression/README.md) |
| **Epoch** | One full pass of training over the whole dataset. | [5](../../chapters/05-linear-regression/README.md) |
| **Chain rule** | The calculus rule for functions inside functions: rates multiply along the chain. Behind backprop. | [8](../../chapters/08-backpropagation/README.md) |
| **Computation graph** | A formula drawn as a network of tiny operations, each with a one-line local derivative. | [8](../../chapters/08-backpropagation/README.md) |
| **Backpropagation** | The algorithm computing every parameter's gradient in one backward sweep over the computation graph. | [8](../../chapters/08-backpropagation/README.md) |
| **Autograd** | An engine that builds the computation graph as code runs, so backprop needs no hand-derived formulas. | [8](../../chapters/08-backpropagation/README.md) |

## Building blocks

| Term | Meaning | Ch. |
|------|---------|-----|
| **Classifier** | A model whose output is a category (spam/not spam) rather than a number. | [6](../../chapters/06-logistic-regression/README.md) |
| **Sigmoid** | The S-shaped function $1/(1+e^{-z})$ squashing any number into (0, 1); turns weighted sums into probabilities. | [6](../../chapters/06-logistic-regression/README.md) |
| **Logistic regression** | The simplest trained classifier: a weighted sum through a sigmoid, trained with cross-entropy. | [6](../../chapters/06-logistic-regression/README.md) |
| **Decision boundary** | The surface where a classifier switches its answer (where its probability crosses the threshold). | [6](../../chapters/06-logistic-regression/README.md) |
| **Artificial neuron** | A weighted sum of inputs plus a bias, through an activation function. The building block of networks. | [7](../../chapters/07-perceptron-and-neurons/README.md) |
| **Activation function** | The nonlinear "bend" on a neuron's weighted sum (step, sigmoid, tanh, ReLU). Without it, layers collapse. | [7](../../chapters/07-perceptron-and-neurons/README.md) |
| **Perceptron** | The 1957 original: one neuron with a step activation and a simple mistake-driven learning rule. | [7](../../chapters/07-perceptron-and-neurons/README.md) |
| **ReLU** | The activation `max(0, z)`: zero for negatives, unchanged for positives. The modern default. | [7](../../chapters/07-perceptron-and-neurons/README.md) |
| **Linearly separable** | A dataset whose two classes split by one straight line. What one neuron can learn; XOR cannot. | [7](../../chapters/07-perceptron-and-neurons/README.md) |
| **Layer** | A group of neurons reading the same inputs in parallel; the inner ones are *hidden* layers. | [7](../../chapters/07-perceptron-and-neurons/README.md) |
| **MLP** | Multi-layer perceptron: the basic network, layers feeding layers. | [9](../../chapters/09-first-neural-network/README.md) |
| **Softmax** | Turns raw scores into probabilities summing to 1. The many-class version of the sigmoid. | [9](../../chapters/09-first-neural-network/README.md) |
| **One-hot** | A label as a vector with a single 1 at the true class (digit 3 → 0,0,0,1,0,…). | [9](../../chapters/09-first-neural-network/README.md) |

## Training in practice

| Term | Meaning | Ch. |
|------|---------|-----|
| **MNIST** | The classic dataset of 70,000 handwritten digits (28×28 grayscale). | [9](../../chapters/09-first-neural-network/README.md) |
| **Batch / mini-batch** | The group of examples in one training step; gradients are averaged over it. | [9](../../chapters/09-first-neural-network/README.md) |
| **SGD** | Stochastic gradient descent: gradient descent on random mini-batches instead of the full dataset. | [9](../../chapters/09-first-neural-network/README.md) |
| **Test set** | Examples held back from training, used only for the final evaluation. | [9](../../chapters/09-first-neural-network/README.md) |
| **Tensor** | An n-dimensional array of numbers (vector=1D, matrix=2D, image batch=4D). Underneath: storage+shape+strides. | [10](../../chapters/10-intro-to-pytorch/README.md) |
| **Broadcasting** | The rule letting tensors of different shapes combine, stretching the smaller one automatically. | [10](../../chapters/10-intro-to-pytorch/README.md) |
| **View / stride** | A tensor reusing another's storage with different shape/strides; reshape/transpose/slice copy nothing. | [10](../../chapters/10-intro-to-pytorch/README.md) |
| **nn.Module** | PyTorch's base class for models; collects all parameters so optimizers and device moves reach them. | [10](../../chapters/10-intro-to-pytorch/README.md) |
| **Overfitting** | Memorizing training data and failing on new data; a growing train/validation accuracy gap. | [11](../../chapters/11-training-deep-networks/README.md) |
| **Momentum** | An optimizer that steps by a running average of recent gradients: consistent directions accelerate. | [11](../../chapters/11-training-deep-networks/README.md) |
| **Adam** | The default modern optimizer: momentum plus per-parameter step normalization. | [11](../../chapters/11-training-deep-networks/README.md) |
| **Validation set** | Data held out and checked *during* training to detect overfitting; distinct from the test set. | [11](../../chapters/11-training-deep-networks/README.md) |
| **Early stopping** | Stop when validation accuracy stops improving. The cheapest defense against overfitting. | [11](../../chapters/11-training-deep-networks/README.md) |
| **Weight decay** | A small penalty pulling weights toward zero, taxing the large weights memorization needs. | [11](../../chapters/11-training-deep-networks/README.md) |
| **Dropout** | Randomly zeroing hidden activations in training so no neuron relies on specific partners. | [11](../../chapters/11-training-deep-networks/README.md) |
| **Batch normalization** | Re-standardizing each layer's activations over the batch, so deep stacks train on stable scales. | [11](../../chapters/11-training-deep-networks/README.md) |
| **He initialization** | Starting weights as noise scaled by √(2/fan-in), keeping signal variance stable through ReLU. | [11](../../chapters/11-training-deep-networks/README.md) |
| **Dataset / DataLoader** | PyTorch's data pieces: a Dataset gives example *i*; a DataLoader shuffles, batches, and transforms. | [12](../../chapters/12-data-pipelines/README.md) |
| **Data augmentation** | Label-preserving random distortions applied fresh each epoch, making the training set effectively infinite. | [12](../../chapters/12-data-pipelines/README.md) |
| **Confusion matrix** | A table of every (true, predicted) class pair; off-diagonal cells show what the model mixes up. | [12](../../chapters/12-data-pipelines/README.md) |
| **Precision / recall** | Of what it called X, how much was X (precision); of all true X, how much it found (recall). | [12](../../chapters/12-data-pipelines/README.md) |
| **GPU** | Hardware doing thousands of simple math operations in parallel; what makes deep learning fast. | [E](../E-hardware-guide/README.md) |

## Language and transformers

| Term | Meaning | Ch. |
|------|---------|-----|
| **Tokenization / BPE** | Splitting text into tokens; byte-pair encoding fuses the most frequent adjacent pair, repeatedly. | [20](../../chapters/20-text-and-tokenization/README.md) |
| **Embedding** | A learned vector for a token (or any discrete item); similar items get similar vectors. | [20](../../chapters/20-text-and-tokenization/README.md) |
| **Recurrent network (RNN)** | A network processing a sequence step by step, carrying a hidden state as memory. LSTMs/GRUs are gated variants. | [21](../../chapters/21-recurrent-networks/README.md) |
| **Attention** | Each position attends to every other by matching queries to keys and blending values. | [22](../../chapters/22-attention-and-transformers/README.md) |
| **Query / key / value** | The three learned vectors per position in attention: what I seek, what I offer, what I hand over. | [22](../../chapters/22-attention-and-transformers/README.md) |
| **Transformer** | The architecture stacking attention + MLP blocks with residuals; the basis of every modern LLM. | [22](../../chapters/22-attention-and-transformers/README.md) |
| **GPT** | A decoder-only transformer with causal (masked) attention, trained to predict the next token. | [23](../../chapters/23-gpt-from-scratch/README.md) |
| **Perplexity** | e^loss for a language model: "as uncertain as a fair choice among this many tokens." Lower is better. | [23](../../chapters/23-gpt-from-scratch/README.md) |
| **Temperature / top-k** | Sampling controls: temperature scales the logits (higher = wilder), top-k keeps only the k likeliest. | [23](../../chapters/23-gpt-from-scratch/README.md) |
| **Checkpoint** | A file saving a model's parameters and training state mid-training, so a run can resume after a stop. | [24](../../chapters/24-train-your-mini-llm/README.md) |
| **LLM** | A network trained to predict the next token of text — enough to write, summarize, and converse. | [24](../../chapters/24-train-your-mini-llm/README.md) |
| **Quantization** | Storing weights in fewer bits (int8) with a scale factor, shrinking a model ~4× with little quality loss. | [25](../../chapters/25-llm-inference-in-c/README.md) |

## Generative and beyond

| Term | Meaning | Ch. |
|------|---------|-----|
| **Autoencoder / VAE** | An encoder-bottleneck-decoder network; the variational version makes the bottleneck sampleable, enabling generation. | [26](../../chapters/26-autoencoders-and-vaes/README.md) |
| **Latent space** | The compact code an encoder maps data into; smooth ones let you sample and interpolate. | [26](../../chapters/26-autoencoders-and-vaes/README.md) |
| **GAN** | Generative adversarial network: a generator and discriminator trained against each other. | [27](../../chapters/27-gans/README.md) |
| **Diffusion model** | A generator learning to reverse gradual noising; generation is repeated denoising from pure noise. | [28](../../chapters/28-diffusion-models/README.md) |
| **Conditioning / guidance scale** | Feeding an instruction (a label or text prompt) into a generator; the scale dials how strongly it obeys. | [29](../../chapters/29-text-to-image-and-video/README.md) |
| **Q-learning / DQN** | Learning the value of each action in each state; a table for small problems, a network (DQN) for large ones. | [30](../../chapters/30-reinforcement-learning/README.md) |
| **RLHF** | Reinforcement learning from human feedback: tuning an LLM to be helpful using human preference rankings as reward. | [30](../../chapters/30-reinforcement-learning/README.md) |
| **Deployment** | Running a trained model in production: export (TorchScript/ONNX), quantize, serve without the training framework. | [31](../../chapters/31-deployment/README.md) |
