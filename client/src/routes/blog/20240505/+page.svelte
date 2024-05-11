<script>
    import predictions from "./predictions.png";
    import { CodeBlock } from "@skeletonlabs/skeleton";

    function navigate(url) {
        window.location.href = url;
    }
</script>

<div
    class="container h-full mx-auto flex justify-center items-center leading-relaxed"
>
    <div class="space-y-5 m-10 custom-container">
        <h1 class="h1 text-center mb-12" on:click={() => navigate("/")}>
            @oj-sec
        </h1>

        <h2 class="h2">
            Evaluating Large Language Models as future event forecasters - Part
            Two: Performance & token sampling
        </h2>
        <p>5 May 2024</p>
        <p class="card variant-filled-ghost p-4">
            You can access a Juypter notebook (built for Colab) associated with
            this post <a
                href="https://github.com/oj-sec/blog/blob/main/notebooks/20240505.ipynb"
                style="text-decoration: underline; color: lightblue;">here</a
            >.
        </p>
        <h3 class="h3">Introduction</h3>
        <p>
            We <a
                href="/blog/20240404"
                style="text-decoration: underline; color: lightblue;"
                >left off</a
            > making predictions by repeatedly invoking a simple prompt and then
            examining the distribution of results. It's worth us examining what was
            going on in a little more detail. We got different results as the result
            of our temperature setting, which was configured to 0.7. Temperature
            in AI models is a measure of how stochastic our outputs are. At a temperature
            of zero, a model will always generate the most likely token, resulting
            in deterministic output. As we increase temperature, token selection
            becomes increasingly random, resulting in more diverse outputs. Temperature
            is commonly set in the range of 0.7-1.0 and higher temperature is often
            perceived by users as higher model intelligence and creativity.
        </p>
        <p>
            We were using temperature to approximate the model's tendency to
            choose particular predictions over many samples. But we can achieve
            the same result more directly, by examining the probabilities of
            options straight from the model. This will give us a performance
            speedup equal to the number of samples we intend to take. While
            optimising a system so early is generally a mistake, its hard to
            leave a straightforward 100x or greater performance improvement on
            the table. To understand how we can peer directly into the model's
            probabilities, we'll need to understand tokenisation.
        </p>
        <h3 class="h3">Tokenisation</h3>
        <p>
            Language models operate on text fragments called tokens, both when
            consuming prompts and generating output. Tokens are semantic
            fragments of written text, typically at the word and sub-word level,
            but sometimes down to the character level. We can tokenise some text
            as an example by directly accessing the tokeniser inside the model
            object maintained for us by Guidance:
        </p>
        <CodeBlock
            language="python"
            code={`# create a text string to tokenise
string = "Mindsets tend to be quick to form but resistant to change."

# generate the tokens by accessing the model tokeniser
tokens_encoded = llm.engine.model_obj.tokenize(str.encode(string))

# decode the tokens
tokens = []
for token in tokens_encoded:
    if token:
        tokens.append(llm.engine.model_obj.detokenize([token]).decode("utf-8", errors="replace"))

# show results
print(tokens_encoded)
print(tokens)`}
        ></CodeBlock>
        <p>We get back the following output:</p>
        <CodeBlock
            language="plaintext"
            code={`[1, 14683, 6591, 6273, 298, 347, 2936, 298, 1221, 562, 605, 11143, 298, 2268, 28723]
['', ' Mind', 'sets', ' tend', ' to', ' be', ' quick', ' to', ' form', ' but', ' res', 'istant', ' to', ' change', '.']`}
        ></CodeBlock>
        <p>
            We can observe a mixture of whole and partial words in the output.
            We can also observe that words commonly have a leading space, which
            may be important for us to account for in some circumstances. A
            common rule of thumb for evaluating how many tokens are present in a
            particular string is that there are approximately four characters
            per token and approximately one token for every 0.75 words.
        </p>
        <h3 class="h3">A note on how Guidance handles tokens</h3>
        <p>
            When we provide Guidance with a regular expression, possible tokens
            are evaluated against the constraining regular expression and
            discarded if they do not match. This functioning is critical for us
            to understand, because we might otherwise assume that the generation
            is evaluated in larger chunks, like whole words or entire phrases.
            This misunderstanding can result in unexpected behaviour due to the
            model starting to generate a coherent answer that is consistent with
            the start of an incoherent option.
        </p>
        <p>
            We can borrow an example from Guidance's Github <a
                href="https://github.com/guidance-ai/guidance/issues/564"
                style="text-decoration: underline; color: lightblue;">issues</a
            >:
        </p>
        <CodeBlock
            language="python"
            code={`from guidance import models, select

# load the model
llm = models.LlamaCpp("./models/mistral-7b-openorca.Q4_K_M.gguf", n_gpu_layers=20, n_ctx=4096) 

# demonstrate bad generation - note that select() functions identically to a regex of the form "(cloud|skill)"
llm + 'A word very similar to "sky" is "' + select(["cloud","skill"])`}
        ></CodeBlock>
        <p>
            Which gives us the perplexing output <span class="pre p-1"
                >"skill"</span
            >
            rather than <span class="pre p-1">"cloud"</span>. The model was
            trying to generate a reasonable answer (<span class="pre p-1"
                >"skies"</span
            >) that collided with the invalid
            <span class="pre p-1">"skill"</span> option. Once the model started
            generating, it could only output
            <span class="pre p-1">"skill"</span> despite the low coherence of the
            answer. As noted by a Guidance contributor, we can address this particular
            case by putting the options directly into the prompt to provide some
            context. We'll cover another pattern for addressing this issue in the
            next part in this series, but it is critical for us to understand that
            Guidance's constraints are at the naive token level so that we don't
            expect contextual evaluation where none is occurring.
        </p>
        <h3 class="h3">Sampling token probabilities directly</h3>
        <p>
            With a background on tokens, we can have a look at directly
            accessing the probability of a particular output. Because generation
            happens token by token, we need to evaluate the probability of each
            token in sequence. These token-specific sequential probabilities
            commonly called the "logprobs" of tokens, defined as log(p) where p
            is the probability of the token occurring given the preceding
            tokens, both generated and prompt. We're going to stick with
            straight probabilities today, but if you shift this paradigm to a
            different stack, including OpenAI APIs, logprobs is the term to look
            for.
        </p>
        <p>
            Unfortunately, Guidance's interface for accessing logits is fairly
            nascent, so we need to implement the method for accessing
            probabilities ourselves. To keep things from getting too complex for
            now, we can cheat by pregenerating the <span class="pre p-1"
                >"0."</span
            > for predictions and instead just evaluating the probabilities of tokens
            in the tenths place. We can access the token logprobs with the following
            code:
        </p>
        <CodeBlock
            language="python"
            code={`from guidance import models, gen
import llama_cpp
import torch
import math
import json

# load the model
llm = models.LlamaCpp("./models/mistral-7b-openorca.Q4_K_M.gguf", compute_log_probs=True, n_gpu_layers=20, n_ctx=4096) 

# define a regular expression to match a single number
output_regex = r"\\d"

# define our prompt - note that we've added a "0." force output to just examine the tenths place
prompt = 'Predict the likelihood of the following outcome on a scale from 0.00 to 1.00, with 0.00 meaning the event is impossible and 1.00 meaning the event is certain to occur: "Donald Trump will win the 2024 US election."\\nPREDICTION:0.'

# run constrained inference - note that we have set temperature to zero
output = llm + prompt + gen(name="response", regex=output_regex, max_tokens=1, temperature=0.0)

# define the options we want to check the probs for
options = [f"{n}" for n in range(0,10)] 

# retrieve the logits from the model object
logits = llama_cpp.llama_get_logits(llm.engine.model_obj.ctx)

# tokenize our options
option_tokens = [llm.engine.model_obj.tokenize(str.encode(o)) for o in options]

# retrieve just the option token, discarding the <s> added by the tokenizer
option_tokens = [o[2] for o in option_tokens] 

# retrieve the logits for the option
option_logits = [logits[o] for o in option_tokens]

# convert the logits into probabilities
option_probs = torch.softmax(torch.tensor(option_logits), dim=0)

# typecast to floats 
option_probs = [float(o) for o in option_probs]

# zip the options and probabilities together
option_probs = dict(zip(options, option_probs))

# get the top token
top_token = max(option_probs, key=option_probs.get)

# print results
print(f"The highest probability option in the tenths place is: {top_token.strip()}")
print("The probability distribution for the tenths place is: ")
print(json.dumps(option_probs, indent=4))`}
        />
        <p>Which gives us the following results:</p>
        <CodeBlock
            language="plaintext"
            code={`The highest probability option in the tenths place is: 2
The probability distribution for the tenths place is: 
{
    "0": 0.15142710506916046,
    "1": 0.15139013528823853,
    "2": 0.18116815388202667,
    "3": 0.16766728460788727,
    "4": 0.12874439358711243,
    "5": 0.12978236377239227,
    "6": 0.0467846542596817,
    "7": 0.02292967587709427,
    "8": 0.010299457237124443,
    "9": 0.009806782938539982
}`}
        />
        <p>
            Here we have shortcut directly to the actual probability
            distribution for the tenths place in our prediction space. In
            effect, this is an instant 100x speedup relative to our previous
            approach of using temperature to iteratively explore this
            distribution by repeated generations. While we've kept things simple
            to demonstrate the concept, this approach can be extrapolated to
            multi-token generation by stepping the model through each token.
        </p>
        <p>
            This is a powerful capability with broad applicability to LLM tasks.
            If we're working on a jailbreak technique, we can evaluate exactly
            how likely it is to occur to properly assess its risk. If we're
            working on classification, we can evaluate the confidence of a given
            prediction. If we're performing finetuning or prompt engineering, we
            get granular insight into whether we're getting hotter or colder as
            we make changes.
        </p>
        <p>
            To finish up, we can repeat our proof of concept showing the actual
            probability distributions for previous proof of concept predictions.
            A little interestingly, the model is less emphatic on both
            predictions than our previous small sample size suggested.
        </p>
        <img src={predictions} alt="Predictions" />
    </div>
</div>

<style>
    @media (min-width: 640px) {
        .custom-container {
            max-width: calc(100% - 4rem);
        }
    }
    @media (min-width: 768px) {
        .custom-container {
            max-width: calc(100% - 6rem);
        }
    }
    @media (min-width: 1024px) {
        .custom-container {
            max-width: calc(100% - 8rem);
        }
    }
    .custom-container {
        padding-left: 8rem;
        padding-right: 8rem;
    }
</style>
