# Why I taught a microcontroller to understand French

*How a keyboard implant, a Teensy MITM, and a LoRa link turned a malware assessment into an embedded NLP project*

During a cybersecurity assessment, we had to design and implement a malware scenario. I did not want a purely software implant. I wanted a physical ingress point that could live in plain sight, so I used a keyboard as the host and hid the whole system inside it.

The setup was simple to describe and much harder to build. A Teensy sat between the original keyboard controller and the computer, acting as a man-in-the-middle on the keystroke path. It received the keystrokes coming from the keyboard electronics, forwarded them so the keyboard still behaved normally for the user, and logged the stream for the implant. Alongside it sat a RAK LoRa board used for outbound communication and remote tasking. The concept was neat. The reality was a lot of soldering, wiring, signal issues, power constraints, and the usual embedded frustration of fitting several boards into a shell that was only ever meant to contain a keyboard.

At that point, though, the hardware was only half the problem. Building a discreet ingress is one thing. Moving data out without being noticed is another.

There is little value in building a hidden implant if the moment it starts talking, it gets noticed by the network. Hiding on a host and hiding on a monitored enterprise network are not the same thing. Proxies, inspection systems, and anomaly detection make ordinary exfiltration channels much harder to trust. So instead of fighting that battle over TCP, I chose a different outbound path: LoRa.

That solved one problem and created the next. LoRa gave me a discreet, long-range, low-power channel that made sense for an embedded implant. It also gave me a severe bandwidth limit. Once your outbound path is narrow, raw exfiltration becomes a bad strategy. You cannot afford to ship everything and decide later what matters. The device has to decide locally what is worth sending.

That constraint is what turned the project from a hardware implant into an embedded NLP problem.

## The actual goal

Once bandwidth became the central constraint, the real task became much clearer. I was not trying to make a microcontroller "understand French" in the broad sense. I only needed it to do one practical thing well: classify short French workplace-style messages into a small set of useful categories so the implant could send only the interesting ones.

The final label set was:

`ACCOUNTING`, `BANKING`, `BUSINESS`, `CYBER`, `GOSSIP`, `HR_COMPLAINT`, `HR_HIRING`, `INFRA`, `LOVE`, `MISC`, `TECH`

This was not an academic benchmark. It was a triage system. A sentence about lunch or scheduling did not deserve the same bandwidth as a sentence about credentials, financial operations, infrastructure failures, or internal conflict.

Once the task was framed that way, the hardware constraints started to matter in a much more concrete way. The target platform was a Teensy 4.1: a 600 MHz Cortex-M7 with about 8 MiB of flash and 1 MiB of RAM. That is generous for a microcontroller and tiny for modern NLP, which meant the design had to stay small, deterministic, and easy to reproduce in C++.

That also changed how I thought about the data. The final cleaned dataset contained 24,842 labeled French samples, but the real work was not chasing architectural novelty. It was finding where the classifier was confused and reshaping the data to address those cases: `TECH` versus `CYBER`, `INFRA` versus `ACCOUNTING`, `HR_COMPLAINT` versus `GOSSIP`, and the always awkward `MISC` class. Under tight constraints, a small model has very little room to waste capacity on noisy labels, generator artifacts, or vague boundaries, so data quality mattered a lot.

Once the task, the hardware budget, and the data constraints were all clear, the next question became straightforward: what kind of pipeline could actually fit?

## The pipeline

The final pipeline was intentionally simple:

1. Normalize the text.
2. Extract symbolic features from it.
3. Hash those features into a fixed-size vector.
4. Feed that vector to a small MLP classifier.

This was the key simplification. Instead of storing a vocabulary and an embedding table, the system used deterministic text processing: lowercase, accent removal, punctuation cleanup, truncation to the first 25 words, then a mix of character n-grams, word unigrams, bigrams, trigrams, and positional markers. Those features were hashed into a vector of length `8192`.

That fixed-size representation solved several problems at once. It kept memory usage predictable, avoided shipping a live vocabulary to the device, and made the Python pipeline reproducible in embedded C++. But to understand why that helps, one idea needs to be very clear first: the idea of a bucket.

## What a bucket actually is

This was the part I found hardest to explain clearly at first.

A phrase is converted into one vector of length `8192`. You can picture it as an array with positions numbered from `0` to `8191`. Each position is a bucket. At the start, every bucket contains `0`.

So yes: each phrase becomes one 8192-dimensional vector. Not 8192 separate vectors, and not a custom number of features that changes with the sentence. Every phrase is projected into the same fixed space.

That is where hashing comes in. We never store a live vocabulary. Instead, every symbolic token produced by the feature extractor is sent through a hash function and assigned to one of those 8192 buckets. Once that idea is clear, the rest of the pipeline becomes much easier to read.

## How a sentence becomes a vector

Take this sentence:

`Le serveur principal est down`

Here is what happens, step by step.

1. **Normalize the text.**  
   Lowercase it, strip accents, remove punctuation, and keep only the first 25 words.  
   Here we get:

   `["le", "serveur", "principal", "est", "down"]`

2. **Generate character-level tokens.**  
   Each word is padded with angle brackets to mark word boundaries.  
   `serveur` becomes `<serveur>`  
   Then the code extracts overlapping character n-grams. Depending on the configured range, that gives tokens such as:

   `C_<se`, `C_ser`, `C_erv`, `C_rve`, `C_veu`, `C_eur`, `C_ur>`

3. **Generate word-level tokens.**  
   The same sentence also produces:

   `W_le`, `W_serveur`, `W_principal`, `W_est`, `W_down`

4. **Generate phrase tokens.**  
   Local context is captured with bigrams and trigrams:

   `B_le_serveur`, `B_serveur_principal`, `B_principal_est`, `B_est_down`

   and possibly:

   `T_le_serveur_principal`, `T_serveur_principal_est`, `T_principal_est_down`

5. **Add position tokens.**  
   The first and last words are marked explicitly:

   `POS_START_le`, `POS_END_down`

At this point, the sentence is no longer just raw text. It has become a collection of symbolic signals. The next step is to place those signals into the fixed vector space.

6. **Hash each symbolic token into a bucket.**  
   Each token is hashed with MurmurHash3, then mapped to one bucket with:

   `j = abs(h(token)) mod 8192`

   If `W_serveur` hashes to `1532`, that means bucket `1532` in the sentence vector will be updated. If `C_ser` hashes to `287`, then bucket `287` will be updated. Those bucket numbers are not meaningful by themselves. What matters is that the mapping is deterministic.

Hashing tells us *where* a token contributes. The next question is *how much* it contributes.

7. **Apply a signed weighted update.**  
   In the current extractor, each token updates the vector with:

   `x[j] += s(token) * w(token)`

   where:

   - `x` is the 8192-dimensional vector
   - `j` is the bucket index
   - `s(token)` is either `+1` or `-1`
   - `w(token)` is the weight of that feature family

   So if `W_serveur` lands in bucket `1532` with positive sign and weight `9.2`, then `x[1532]` becomes `+9.2`. If `C_ser` lands in bucket `287` with negative sign and weight `3.1`, then `x[287]` becomes `-3.1`.

8. **Repeat for every token in the sentence.**  
   When the process ends, most buckets are still zero. Only a small subset has been touched. That final sparse vector is what the classifier sees.

This is the important intuition: the word `serveur` does not become one coordinate. It becomes a small cloud of contributions spread across the vector through character fragments, word tokens, phrase tokens, and sometimes positional markers.

If two different tokens land in the same bucket, that is a collision. It sounds bad, but it is the trade-off that keeps the representation small enough to deploy. With enough buckets and useful feature families, collisions behave like manageable noise rather than a complete failure.

Once the mechanics of the vector are clear, one more part of the design starts to matter: not every token family should contribute equally.

## Why the weights matter

Not all token families carry the same kind of information.

In practice, word-level tokens tend to carry the strongest weight, character n-grams help with noisy spelling and variations, and bigrams or trigrams add local context. The result is not just a bag of hashed tokens. It is a weighted combination of several views of the same sentence.

That matters because it explains why the system holds up on short, messy text. If the exact word changes slightly, the character n-grams still carry signal. If the wording is clean, the unigrams and phrase tokens give stronger semantic anchors. All of that ends up in the same fixed vector.

At that stage, the remaining problem is no longer feature design in principle. It is parameter tuning in practice.

## Where Optuna came in

There were too many interacting parameters to tune by hand. The feature extractor had its own knobs, such as the relative weights of word tokens, character n-grams, bigrams, and trigrams. The classifier added more choices with hidden-layer sizes, activation function, regularization, and learning rate. Changing one of these could affect the usefulness of the others, which made manual tuning slow and unreliable.

That is where Optuna came in.

Optuna is a hyperparameter optimization framework. In simple terms, it automates the part of machine learning where you would otherwise spend hours changing parameters by hand, retraining, checking the results, and repeating the process. Instead of guessing which combination of settings might work best, you define a search space, for example the weight of character n-grams, the size of the hidden layers, or the learning rate, and Optuna runs many training trials to explore that space for you.

For this project, that meant turning parameter tuning into a structured search. Each trial trained a candidate configuration, measured its performance, and used that result to guide the next trials toward more promising regions of the space.

It tuned:

- the weights of each feature family
- the minimum and maximum character n-gram sizes
- the hashed input size
- the two hidden-layer widths of the MLP
- the activation function
- the regularization strength
- the learning rate

The objective was not just raw accuracy. In a multi-class system like this, raw accuracy can hide a weak class. So the search favored more balanced behavior: balanced accuracy, mean recall, minimum class recall, and resistance to overfitting.

That matters because a search procedure is only useful if it produces a model that is not just accurate on average, but actually usable under the original hardware constraints.

## What it achieved

The most interesting compact checkpoint in the workspace used an `8192`-dimensional input with hidden layers of `48` and `40`. That gave:

- `395,675` parameters
- about `1.51 MiB` of float32 weights
- `91.65%` balanced accuracy
- `84.06%` recall on the weakest class

On a Teensy 4.1, that is a workable footprint. The main constraint is flash, not RAM. Even with a dense `8192`-float input buffer, runtime memory stays in the tens of kilobytes, while the stored weights take the larger share of the budget.

There is also an INT8 export path in the refactored code. If it holds up in end-to-end validation, that same model could drop from about `1.51 MiB` to roughly `0.38 MiB`.

Those numbers bring the story back to where it started. The point was never to put AI on a microcontroller for the sake of it. The point was to solve a very physical problem: a hidden implant with a narrow covert link needed to decide locally which text was worth the cost of transmission.

## What I took from it

That is what I like about this project. The classifier was not there because it was fashionable. It was there because bandwidth made it the cleanest solution.

In the end, this was less a story about model ambition than about fitting the right amount of intelligence into the right place. A hidden keyboard implant with a LoRa channel did not need a general language model. It needed a small on-device filter that could decide which messages were worth sending.

That is exactly what this project became.
