# Appendix D — Glossary

Every AI term used in this course, defined in one or two plain sentences. Terms link to the chapter that explains them fully. This appendix grows as the course grows.

**Artificial intelligence (AI)** — the broad field of making computers do things that seem to need intelligence. Machine learning is the part of AI this course teaches. → [Chapter 1](../../chapters/01-what-is-ai/README.md)

**Machine learning (ML)** — writing programs that improve from data instead of being told every rule. → [Chapter 1](../../chapters/01-what-is-ai/README.md)

**Deep learning** — machine learning using neural networks with many layers. → [Chapter 1](../../chapters/01-what-is-ai/README.md)

**Model** — the thing being trained: a function with adjustable numbers (parameters) inside. → [Chapter 1](../../chapters/01-what-is-ai/README.md)

**Parameter / weight** — one of the adjustable numbers inside a model. Training means finding good values for them. → [Chapter 5](../../chapters/05-linear-regression/README.md)

**Bias (parameter)** — the additive parameter in a neuron or linear model; shifts the output up or down independent of the inputs. → [Chapter 5](../../chapters/05-linear-regression/README.md)

**Feature** — one measurable property of an input example (a pixel brightness, a house's size). → [Chapter 1](../../chapters/01-what-is-ai/README.md)

**Label** — the correct answer attached to a training example. → [Chapter 1](../../chapters/01-what-is-ai/README.md)

**Supervised learning** — learning from examples that come with labels. → [Chapter 1](../../chapters/01-what-is-ai/README.md)

**Unsupervised learning** — learning structure from data without labels. → [Chapter 1](../../chapters/01-what-is-ai/README.md)

**Reinforcement learning** — learning by acting and receiving rewards instead of labeled examples. → [Chapter 30](../../chapters/30-reinforcement-learning/README.md)

**Training** — the process of adjusting a model's parameters to make its predictions better on data. → [Chapter 5](../../chapters/05-linear-regression/README.md)

**Inference** — using an already-trained model to make predictions. → [Chapter 1](../../chapters/01-what-is-ai/README.md)

**Loss function** — a formula that scores how wrong the model currently is, as one number. Training pushes this number down. → [Chapter 5](../../chapters/05-linear-regression/README.md)

**Mean squared error (MSE)** — a loss for numeric predictions: average of (prediction − truth)². → [Chapter 5](../../chapters/05-linear-regression/README.md)

**Gradient** — the vector of slopes of the loss with respect to each parameter; points in the direction of steepest increase. → [Chapter 3](../../chapters/03-derivatives-and-gradients/README.md)

**Gradient descent** — the core training algorithm: repeatedly nudge every parameter a small step against its gradient. → [Chapter 5](../../chapters/05-linear-regression/README.md)

**Learning rate** — the step size of gradient descent. Too big diverges, too small crawls. → [Chapter 5](../../chapters/05-linear-regression/README.md)

**Epoch** — one full pass of training over the whole dataset. → [Chapter 5](../../chapters/05-linear-regression/README.md)

**Vector** — a list of numbers. → [Chapter 2](../../chapters/02-vectors-and-matrices/README.md)

**Matrix** — a grid of numbers. Neural network layers are mostly matrix multiplications. → [Chapter 2](../../chapters/02-vectors-and-matrices/README.md)

**Dot product** — multiply two vectors element by element and sum: the basic "how aligned are these?" operation. → [Chapter 2](../../chapters/02-vectors-and-matrices/README.md)

**Tensor** — the general word for an n-dimensional array of numbers (vector = 1D, matrix = 2D, an image batch = 4D). Underneath: flat storage + shape + strides. → [Chapter 10](../../chapters/10-intro-to-pytorch/README.md)

**Broadcasting** — the rule that lets tensors of different shapes combine: the smaller shape is stretched automatically (a vector added to a matrix applies to every row). → [Chapter 10](../../chapters/10-intro-to-pytorch/README.md)

**View / stride** — a tensor that reuses another tensor's storage with different shape/strides; reshape, transpose, and slicing are views, so they copy nothing. → [Chapter 10](../../chapters/10-intro-to-pytorch/README.md)

**nn.Module** — PyTorch's base class for models: it collects all parameters inside so optimizers and device moves can reach them. → [Chapter 10](../../chapters/10-intro-to-pytorch/README.md)

**Probability distribution** — an assignment of probabilities to all possible outcomes, summing to 1. → [Chapter 4](../../chapters/04-probability-basics/README.md)

**Expected value** — the long-run average of a random quantity. → [Chapter 4](../../chapters/04-probability-basics/README.md)

**Cross-entropy** — a loss for classifiers that measures how surprised the model is by the true answers. Low surprise = good model. → [Chapter 4](../../chapters/04-probability-basics/README.md)

**Classifier** — a model whose output is a category (spam/not spam, cat/dog) rather than a number. → [Chapter 6](../../chapters/06-logistic-regression/README.md)

**Sigmoid** — the S-shaped function $1/(1+e^{-z})$ that squashes any number into (0, 1); turns weighted sums into probabilities. → [Chapter 6](../../chapters/06-logistic-regression/README.md)

**Logistic regression** — the simplest trained classifier: a weighted sum passed through a sigmoid, trained with cross-entropy. → [Chapter 6](../../chapters/06-logistic-regression/README.md)

**Decision boundary** — the surface where a classifier switches its answer; where its probability crosses the threshold (usually 0.5). → [Chapter 6](../../chapters/06-logistic-regression/README.md)

**Hyperparameter** — a knob the *person* chooses rather than the model learning it (learning rate, number of epochs, layer sizes). → [Chapter 3](../../chapters/03-derivatives-and-gradients/README.md)

**Artificial neuron** — a weighted sum of inputs plus a bias, passed through an activation function. The building block of neural networks. → [Chapter 7](../../chapters/07-perceptron-and-neurons/README.md)

**Activation function** — the nonlinear "bend" applied to a neuron's weighted sum (step, sigmoid, tanh, ReLU). Without it, stacked layers collapse into one line. → [Chapter 7](../../chapters/07-perceptron-and-neurons/README.md)

**Perceptron** — the 1957 original: one neuron with a step activation and a simple mistake-driven learning rule. → [Chapter 7](../../chapters/07-perceptron-and-neurons/README.md)

**ReLU** — the activation `max(0, z)`: zero for negative inputs, unchanged for positive ones. The default in modern deep networks. → [Chapter 7](../../chapters/07-perceptron-and-neurons/README.md)

**Linearly separable** — a dataset whose two classes can be split by one straight line (or flat surface). What a single neuron can learn; XOR is the classic counterexample. → [Chapter 7](../../chapters/07-perceptron-and-neurons/README.md)

**Layer** — a group of neurons that all read the same inputs and compute in parallel; networks stack layers, and the inner ones are called *hidden* layers. → [Chapter 7](../../chapters/07-perceptron-and-neurons/README.md)

**Chain rule** — the calculus rule for functions inside functions: rates multiply along the chain. The single math fact behind backpropagation. → [Chapter 8](../../chapters/08-backpropagation/README.md)

**Computation graph** — a formula drawn as a network of tiny operations, each with a one-line local derivative. → [Chapter 8](../../chapters/08-backpropagation/README.md)

**Backpropagation** — the algorithm that computes every parameter's gradient in one backward sweep over the computation graph, by applying each node's local rule to the gradient flowing in from above. → [Chapter 8](../../chapters/08-backpropagation/README.md)

**Autograd (automatic differentiation)** — an engine that builds the computation graph as a side effect of running the code, so backpropagation needs no hand-derived formulas. PyTorch's core feature. → [Chapter 8](../../chapters/08-backpropagation/README.md)

**MNIST** — the classic dataset of 70,000 handwritten digits (28×28 grayscale), the traditional first dataset for image models. → [Chapter 9](../../chapters/09-first-neural-network/README.md)

**MLP (multi-layer perceptron)** — the basic neural network: layers of neurons where each layer's outputs feed the next layer's inputs. → [Chapter 9](../../chapters/09-first-neural-network/README.md)

**Softmax** — turns a list of raw scores into probabilities that sum to 1 (exponentiate, then divide by the total). The many-class version of the sigmoid. → [Chapter 9](../../chapters/09-first-neural-network/README.md)

**One-hot** — a label written as a vector with a single 1 at the true class and 0 elsewhere (digit 3 → 0,0,0,1,0,0,0,0,0,0). → [Chapter 9](../../chapters/09-first-neural-network/README.md)

**Batch / mini-batch** — the group of examples processed in one training step; gradients are averaged over it. → [Chapter 9](../../chapters/09-first-neural-network/README.md)

**SGD (stochastic gradient descent)** — gradient descent on random mini-batches instead of the full dataset: noisier steps, many more of them per pass. → [Chapter 9](../../chapters/09-first-neural-network/README.md)

**Test set** — examples held back from training and used only for evaluation; accuracy on it is the only honest measure of a model. → [Chapter 9](../../chapters/09-first-neural-network/README.md)

**Overfitting** — when a model memorizes its training data and fails on new data; visible as a growing gap between training and validation accuracy. → [Chapter 11](../../chapters/11-training-deep-networks/README.md)

**Momentum** — an optimizer upgrade that steps by a running average of recent gradients instead of the current one: consistent directions accelerate, zigzags cancel. → [Chapter 11](../../chapters/11-training-deep-networks/README.md)

**Adam** — the default modern optimizer: momentum plus per-parameter step normalization by the running average of squared gradients. → [Chapter 11](../../chapters/11-training-deep-networks/README.md)

**Validation set** — data held out from training, checked *during* training to detect overfitting and tune knobs; distinct from the test set, which is touched only once at the end. → [Chapter 11](../../chapters/11-training-deep-networks/README.md)

**Early stopping** — stop training when validation accuracy stops improving. The cheapest defense against overfitting. → [Chapter 11](../../chapters/11-training-deep-networks/README.md)

**Weight decay** — a small penalty pulling all weights toward zero during training, taxing the large specific weights memorization needs. → [Chapter 11](../../chapters/11-training-deep-networks/README.md)

**Dropout** — randomly zeroing hidden activations during training so no neuron can depend on specific partners; switched off at evaluation. → [Chapter 11](../../chapters/11-training-deep-networks/README.md)

**Batch normalization** — re-standardizing each layer's activations over the batch (with two learned parameters per neuron), so deep stacks train on stable scales. → [Chapter 11](../../chapters/11-training-deep-networks/README.md)

**He initialization** — starting weights as random noise scaled by √(2/fan-in), keeping signal variance stable through ReLU layers. → [Chapter 11](../../chapters/11-training-deep-networks/README.md)

**Dataset / DataLoader** — PyTorch's two data pieces: a Dataset answers "how many examples, give me example i"; a DataLoader shuffles, batches, and transforms them each epoch. → [Chapter 12](../../chapters/12-data-pipelines/README.md)

**Data augmentation** — label-preserving random distortions (shifts, rotations, crops) applied fresh each epoch, so the training set is effectively infinite. → [Chapter 12](../../chapters/12-data-pipelines/README.md)

**Confusion matrix** — a table counting every (true class, predicted class) pair; the off-diagonal cells show which classes the model mixes up. → [Chapter 12](../../chapters/12-data-pipelines/README.md)

**Precision / recall** — of everything the model called class X, how much really was (precision); of all true X, how much it found (recall). The honest metrics for imbalanced classes. → [Chapter 12](../../chapters/12-data-pipelines/README.md)

**GPU (graphics processing unit)** — hardware that performs thousands of simple math operations in parallel; what makes deep learning fast. → [Appendix E](../E-hardware-guide/README.md)

**Checkpoint** — a file saving a model's parameters (and training state) mid-training, so training can resume after a stop. → [Chapter 24](../../chapters/24-train-your-mini-llm/README.md)

**LLM (large language model)** — a neural network trained to predict the next token of text, which turns out to be enough to write, summarize, and converse. → [Chapter 24](../../chapters/24-train-your-mini-llm/README.md)

**Tokenization / BPE** — splitting text into tokens; byte-pair encoding builds the token vocabulary by repeatedly fusing the most frequent adjacent pair. → [Chapter 20](../../chapters/20-text-and-tokenization/README.md)

**Embedding** — a learned vector representing a token (or any discrete item); similar items end up with similar vectors. → [Chapter 20](../../chapters/20-text-and-tokenization/README.md)

**Recurrent network (RNN)** — a network that processes a sequence step by step, carrying a hidden state as memory. LSTMs and GRUs are gated variants. → [Chapter 21](../../chapters/21-recurrent-networks/README.md)

**Attention** — the mechanism where each position attends to every other by matching queries against keys and blending values; the core of transformers. → [Chapter 22](../../chapters/22-attention-and-transformers/README.md)

**Query / key / value** — the three learned vectors per position in attention: what I seek, what I offer, what I hand over. → [Chapter 22](../../chapters/22-attention-and-transformers/README.md)

**Transformer** — the architecture stacking attention + MLP blocks with residuals; the basis of every modern LLM. → [Chapter 22](../../chapters/22-attention-and-transformers/README.md)

**GPT (decoder-only transformer)** — a transformer with causal (masked) attention trained to predict the next token. → [Chapter 23](../../chapters/23-gpt-from-scratch/README.md)

**Perplexity** — e^loss for a language model: "as uncertain as a fair choice among this many tokens." Lower is better. → [Chapter 23](../../chapters/23-gpt-from-scratch/README.md)

**Temperature / top-k** — sampling controls: temperature scales the logits (higher = wilder), top-k restricts sampling to the k most likely tokens. → [Chapter 23](../../chapters/23-gpt-from-scratch/README.md)

**Quantization** — storing weights in fewer bits (e.g. int8) with a scale factor, shrinking a model ~4× with little quality loss. → [Chapter 25](../../chapters/25-llm-inference-in-c/README.md)
