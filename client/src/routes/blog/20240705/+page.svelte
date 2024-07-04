<script>
    import diagram from "./diagram.png";
    import map1 from "./map1.png";
    import map2 from "./map2.png";
    import map3 from "./map3.png";
    import map4 from "./map4.png";
    import { CodeBlock } from "@skeletonlabs/skeleton";
    import { getModalStore } from "@skeletonlabs/skeleton";

    let modalStore = getModalStore();

    function navigate(url) {
        window.location.href = url;
    }

    function triggerImageModal(image) {
        console.log("image", image);
        const modal = {
            image: image,
            modalClasses:
                "max-w-[90%] max-h-[90%] rounded-container-token overflow-hidden shadow-xl",
        };
        modalStore.trigger(modal);
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
            Research aside - Hallucination detection & LLM explainability
        </h2>
        <p>5 July 2024</p>
        <p class="card variant-filled-ghost p-4">
            You can access a Juypter notebook (built for Colab) associated with
            this post <a
                href="https://github.com/oj-sec/blog/blob/main/notebooks/20240705.ipynb"
                style="text-decoration: underline; color: lightblue;">here</a
            >.
        </p>
        <h3 class="h3">Introduction</h3>
        <p>
            The term "hallucination" is commonly used to describe factually
            inaccurate or incoherent outputs from generative AI models. Large
            Language Models (LLMs) are particularly prone to problematic
            hallucinations because instruct and chat fine tuning bias models
            towards producing confident-looking outputs and models lack any
            ability to introspect about confidence. Instruct and chat fine
            tuning are processes that convert a base LLM, which functions as an
            autocomplete engine that expands on a user's input, to the AI
            assistant, prompt-response/Q&A style outputs we're familiar with.
        </p>
        <p>
            Hallucinations can be a major barrier to deploying AI systems that
            can function without close human oversight and are an important
            consideration for almost every real-world LLM application. In this
            blog, I'm going to dive into hallucination detection and LLM
            explainability. This blog will involve examination of internal model
            states not made available through most inference backends, so we'll
            be using Transformers rather than our usual Guidance+LLama.cpp
            stack. The notebook and code samples provided will not interoperate
            with previously provided code.
        </p>
        <h3 class="h3">Detecting hallucinated outputs</h3>
        <p>
            The idea that any given AI generated sequence can be classified on a
            binary between hallucinatory or factual is an oversimplification.
            The meaning encoded by language models can be thought of as a
            measure of the strength of the associativity between items in a
            sequence. There is no underlying concept of truth, it's all a
            spectrum of likelihoods - given this input, what is the likelihood
            of this output? In the real world, there are degrees of
            incorrectness that have important nuance - for example, an incorrect
            extrapolation from accurately-regurgitated facts probably poses a
            different risk than an outright fabrication.
        </p>
        <div class="flex justify-center mt-0">
            <img src={diagram} alt="Diagram" />
        </div>
        <p>
            LLMs' foundation in likelihoods gives us a clear path towards
            identifying hallucinations. We can assume that a coherent model will
            be less likely to generate a hallucination than a fact.
            Anthropomorphising, we can think of using the likelihood as a proxy
            to evaluate how "sure" of an answer the model is. This might not
            always map to factualness, but in my opinion, knowing that an LLM
            has a low confidence in its output is actually more useful. While a
            low confidence output might be factually correct by chance, we can
            infer that we've pushed the LLM to the limits of its knowledge and
            that we need to exercise caution around that topic. This method
            won't help us find very confident falsehoods, such as when a LLM was
            trained on a common misconception, but that class of hallucination
            is likely to be almost impossible to control for at inference time
            once it has been baked into a model during training.
        </p>
        <p>
            In June 2024, Oxford University researchers published an article in <i
                >Nature</i
            >
            titled
            <a
                href="https://www.nature.com/articles/s41586-024-07421-0"
                style="text-decoration: underline; color: lightblue;"
                >"Detecting hallucinations in large language models using
                semantic entropy"</a
            >
            centered around this idea. The researchers proposed sampling outputs
            repeatedly and evaluating the spread (entropy) of outputs, with an added
            step of bucketing semantically similar answers. Combined with a system
            of extracting and checking standalone facts in larger generations, the
            proposed system had state-of-the-art confabulation detection rates in
            the 0.75-0.80 range depending on model.
        </p>
        <p>
            Solutions that involve repeated inference sampling are extremely
            costly, potentially prohibitively costly. We demonstrated in <a
                href="/blog/20240505"
                style="text-decoration: underline; color: lightblue;"
                >the last blog</a
            >
            that we can substitute sampling with accessing logits and evaluating
            how likely each token is to be generated - the same idea theoretically
            holds here. Accessing logits under Transformers looks like this (note
            that we're using a Python object to encapsulate a bit more functionality
            than usual - check the
            <a
                href="https://github.com/oj-sec/blog/blob/main/notebooks/20240705.ipynb"
                style="text-decoration: underline; color: lightblue;"
                >notebook</a
            > to see the full setup):
        </p>
        <CodeBlock
            language="python"
            code={`
    def generate_with_probability(
        self,
        prompt,
        response_prefix=None,
        max_tokens=200,
        temperature=0.0,
        round_to=6,
    ):
        """
        Method to generate a response with information about
        token probabilities.

        Args:
            prompt: str: The prompt to generate a response for.
            response_prefix: str: String to prefix the LLM's response.
            max_tokens: int: The maximum number of tokens to generate.
            temperature: float: The temperature to use for sampling.
            round_to: int: The number of decimal places to round to.

        Returns:
            str: The generated response.
            float: The total probability of the generated response.
            float: The average probability of each token in the generated response.
            list[tuple]: The individual token probabilities.
        """
        formatted_prompt = f"{
            self.system_prompt_start}{
            self.system_prompt}{
            self.prompt_suffix}{
                self.user_prompt_start}{prompt}{
                    self.prompt_suffix}{
                        self.assistant_prompt_start}"
        if response_prefix:
            formatted_prompt = formatted_prompt + response_prefix
        logging.info("LLMAgent generating response with probability for: %s.", prompt)
        inputs = self.processor(formatted_prompt, return_tensors="pt").to(self.device)
        generate_output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=bool(temperature > 0.0),
            temperature=temperature,
        )
        generate_ids = generate_output.sequences[:, inputs["input_ids"].shape[1] :]
        generated_sequence = generate_ids[0].cpu().numpy()
        generated_sequence = generated_sequence[:-2]  # Remove eos tokens
        response = self.processor.batch_decode(
            [generated_sequence],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        logging.info("  LLMAgent generated response: '%s'", response)
        logits = torch.stack(generate_output.scores, dim=1).cpu()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
        log_likelihood_for_gen = sum(
            log_probs[0, i, token_id].item()
            for i, token_id in enumerate(generated_sequence)
        )
        total_probability_for_gen = round(np.exp(log_likelihood_for_gen), round_to)
        individual_token_probs = []
        for i, token_id in enumerate(generated_sequence):
            token_prob = np.exp(log_probs[0, i, token_id].item())
            individual_token_probs.append(
                (self.processor.decode(token_id), round(token_prob, round_to))
            )
        average_token_propabiility = round(
            sum(token_prob for _, token_prob in individual_token_probs)
            / len(individual_token_probs),
            round_to,
        )
        return (
            response,
            total_probability_for_gen,
            average_token_propabiility,
            individual_token_probs,
        )`}
        ></CodeBlock>
        <p>Let's instantiate the object and run a simple prompt:</p>
        <CodeBlock
            language="python"
            code={`lm = LLMAgent(
    system_prompt="You are a helpful AI assistant that provides correct information as concisely as possible.\\n"
)
response, total_probability, average_token_probability, individual_token_probs = lm.generate_with_probability(
    "What is the capital of Australia?"
)

print(f"Response generated:\\n{response}")
print(f"Total response probability:\\n{total_probability}")
print(f"Average token probability:\\n{average_token_probability}")
print("Individual token probabilities:")
for token in individual_token_probs:
    buff = 20 * " "
    buff = token[0] + buff[len(token[0]) :]
    print(f"{buff} : {token[1]}")`}
        ></CodeBlock>
        <p>Which gives us the following output:</p>
        <CodeBlock
            language="plaintext"
            code={`Response generated:
The capital of Australia is Canberra.
Total response probability:
0.841735
Average token probability:
0.982328
Individual token probabilities:
The                  : 0.845985
capital              : 0.999886
of                   : 0.997526
Australia            : 0.999978
is                   : 0.999964
Can                  : 0.999985
ber                  : 1.0
ra                   : 0.999999
.                    : 0.997632`}
        ></CodeBlock>
        <p>
            We inherit some additional complexity when we extend this idea over
            a whole string rather than just looking at a single token. Our
            lowest confidence token was the starting token, <span
                class="pre p-1">"The"</span
            >, which is is unlikely to be material to the meaning of the answer.
            It's also axiomatic that we're going to get smaller and smaller
            probabilities the longer our string gets.
        </p>
        <p>
            Ultimately, what we want is a single metric than can give us a
            consistent read on confidence, regardless of the length and
            complexity of the input. I included a return type that takes the
            average of the individual tokens rather than their multiple to help
            control for variance in output length. I gave the metrics a test
            over a small sample of the TriviaQA dataset, one of the datasets
            used by the Oxford researchers.
        </p>
        <p>
            Both metrics had comparable performance to identify confabulations,
            in the state of the art range of 0.75-0.80 (noting a small sample
            size). I used 0.5 as the confidence threshold for the overall
            probability, and 0.75 as the confidence threshold for average token
            probability. If an answer fell below the confidence threshold, I
            regarded it as likely wrong, if above the threshold likely correct.
            Taking the average of both metrics eked out a little extra
            performance.
        </p>
        <div
            class="container h-full mx-auto flex justify-center items-center leading-relaxed"
        >
            <div class="space-y-2 m-3 custom-container">
                <div class="table-container">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Method</th>
                                <th>Threshold</th>
                                <th>Result</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Total probability</td>
                                <td class="text-center">0.5</td>
                                <td class="text-center">0.78</td>
                            </tr>
                            <tr>
                                <td>Average token probability</td>
                                <td class="text-center">0.75</td>
                                <td class="text-center">0.76</td>
                            </tr>
                            <tr>
                                <td>Average of both metrics</td>
                                <td class="text-center">0.625</td>
                                <td class="text-center">0.81</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <p>
            These thresholds won't work for generations that aren't
            one-sentence/one-fact, but given that multi-fact generations almost
            certainly need to be decomposed to be evaluated, I think this is a
            performance friendly alternative to the semantic entropy sample and
            bucket technique and worth putting in the toolbox.
        </p>
        <h3 class="h3">Diving deeper</h3>
        <p>
            Like all complex AI architectures, transformer-based models work by
            passing inputs through a series of hidden layers containing a
            network of neurons that activate to various degrees based features
            learned through training. The most advanced public research on LLM
            explainability is Anthropic's incredible
            <a
                href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html"
                style="text-decoration: underline; color: lightblue;"
                >"Mapping the Mind of a Large Language Model"</a
            >
            research, which demonstrates that LLMs coalesce human-comprehensible
            concepts in the activation states of hidden layers. These identifiable
            features can be modulated manually to change a model's behavior.
        </p>
        <p>
            This evokes some interesting questions around the topic of
            hallucinations. Is there a feature hidden in LLMs that is highly
            correlated with hallucinating? In humans, lying and creativity are
            both distinctive brain states. Is there a feature that is
            responsible for influencing the output "I don't know" that we could
            tune up to decrease confabulations?
        </p>
        <p>
            Unfortunately, Anthropic's research involved training custom sparse
            autoencoders that are beyond me to develop and train. We can get a
            glimpse into the internal state of models by extracting the hidden
            layers and visualising activations. We can access the hidden state
            through transformers by instantiating our model with
            <span class="pre p-1">output_hidden_states=True</span> and calling
            <span class="pre p-1">generate()</span> with
            <span class="pre p-1">return_dict_in_generate=True</span>. This
            gives us access to a <span class="pre p-1">hidden_states</span> array
            attribute on our generate output, where the first element is a tensor
            containing the hidden states for the entire context and each subsequent
            array element is a tensor containing the hidden state activations for
            each generated token. I visualised the average activations across the
            entire layer for each token with the below code:
        </p>
        <CodeBlock
            language="python"
            code={`    def visualise_average_activations(self, outputs):
    """
    Method to visualise average activations per layer as a heatmap.
    """
    logging.info("LLMAgent visualising average activations for sequence.")
    tokens = [
        self.processor.decode(input_token) for input_token in outputs.sequences[0]
    ]
    average_activations = []
    for layer_states in outputs.hidden_states[0]:
        avg_activation = layer_states.squeeze(0).mean(dim=-1)
        average_activations.append(avg_activation)

    for layer_states in outputs.hidden_states[1:]:
        for i, layer_state in enumerate(layer_states):
            avg_activation = layer_state.squeeze(0).mean(dim=-1)
            average_activations[i] = torch.cat(
                [average_activations[i], avg_activation]
            )

    average_activations = torch.stack(average_activations, dim=1)
    figsize_x = max(12, len(outputs.hidden_states[0]) * 0.8)
    figsize_y = max(8, len(tokens) * 0.3)

    plt.figure(figsize=(figsize_x, figsize_y))
    sns.heatmap(
        average_activations.detach().cpu().numpy(),
        cmap="mako_r",
        xticklabels=[f"Layer {i}" for i in range(len(outputs.hidden_states[0]))],
        yticklabels=tokens,
        linecolor="lightgrey",
        linewidths=0.2,
        cbar=True,
    )
    plt.title("Average activation per layer per token")
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_bytes = buffer.read()
    plt.close()
    logging.info("LLMAgent visualised average activations for sequence.")
    return image_bytes`}
        />
        <p>Which produces outputs that look like this:</p>

        <div class="text-center">
            <h4 class="h4">
                "What is the capital of Australia?" - Factual generation
            </h4>
        </div>
        <div
            class="flex justify-center mt-0"
            on:click={() => triggerImageModal(map1)}
        >
            <img src={map1} alt="Map 1" />
        </div>
        <div class="text-center">
            <h4 class="h4">
                "What is the target of Sotorasib?" - Incomplete factual
                generation
            </h4>
        </div>
        <div
            class="flex justify-center mt-0"
            on:click={() => triggerImageModal(map2)}
        >
            <img src={map2} alt="Map 2" />
        </div>
        <div class="text-center">
            <h4 class="h4">
                "Which was the first European country to abolish capital
                punishment?" - Confabulation
            </h4>
        </div>
        <div
            class="flex justify-center mt-0"
            on:click={() => triggerImageModal(map3)}
        >
            <img src={map3} alt="Map 3" />
        </div>
        <div class="text-center">
            <h4 class="h4">
                "How did Jock die in Dallas?" - Safety-related refusal
            </h4>
        </div>
        <div
            class="flex justify-center mt-0"
            on:click={() => triggerImageModal(map4)}
        >
            <img src={map4} alt="Map 4" />
        </div>

        <p>
            We can make a few observations. Firstly, we have a very consistent
            average model states until the later layers, where things get more
            diverse. Secondly, we have very distinctive stripes of low average
            activation on punctuation and definite articles. A possible
            explanation for this is that these are indistinct "filler" tokens
            that do not require much activity within the model, but we'd really
            need to look at the individual states to understand further. At this
            resolution we can't see visual patterns that differentiate confident
            answers, confabulations and our refusal, but this isn't particularly
            surprising. We'd need to visualise neuron in each layer individually
            to see such subtle features. Each layer in the tiny
            Phi-3-mini-4k-instruct model I'm using has 3072 neurons for each of
            its 33 layers, so our odds of just eyeballing the features we're
            looking for are next to zero - we'd need to approach it as a
            statistical or deep learning task. Given the length of this blog
            already, that will have to be a topic for the future.
        </p>
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
