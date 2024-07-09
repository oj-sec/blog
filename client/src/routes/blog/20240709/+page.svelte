<script>
    import map1 from "./map1.png";
    import isometricStack from "./isometric_stack.png";
    import map2 from "./map2.png";
    import map3 from "./map3.png";
    import map4 from "./map4.png";
    import map5 from "./map5.png";
    import map6 from "./map6.png";
    import { CodeBlock } from "@skeletonlabs/skeleton";
    import { getModalStore } from "@skeletonlabs/skeleton";
    import TagList from "$lib/tagList.svelte";
    import { Accordion, AccordionItem } from "@skeletonlabs/skeleton";

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

    let tags = [
        "technical",
        "deep-dive",
        "advanced",
        "research",
        "llm",
        "hallucination",
        "explainability",
    ];
</script>

<div
    class="container h-full mx-auto flex justify-center items-center leading-relaxed"
>
    <div class="space-y-5 m-10 custom-container">
        <h1 class="h1 text-center mb-12" on:click={() => navigate("/")}>
            @oj-sec
        </h1>

        <h2 class="h2">
            Research aside - Hallucination detection & LLM explainability - Part
            2
        </h2>
        <p>8 July 2024 - 11 min read</p>
        <TagList {tags} />
        <p class="card variant-filled-ghost p-4">
            You can access a Juypter notebook (built for Colab) associated with
            this post <a
                href="https://github.com/oj-sec/blog/blob/main/notebooks/20240709.ipynb"
                style="text-decoration: underline; color: lightblue;">here</a
            >.
        </p>
        <h3 class="h3">Introduction</h3>
        <p>
            In the <a
                href="/blog/20240505"
                style="text-decoration: underline; color: lightblue;"
                >last blog</a
            > I looked into hallucination detection in Large Language Models, demonstrating
            that we can approach state-of-the-art hallucination detection in a highly
            performant manner by evaluating the model's likelihood of generating
            a given response. I left off posing the question - is there some feature
            buried in the LLM's hidden states, effectively its brain, that is distinctively
            associated with hallucinations?
        </p>
        <p>
            There are some interesting implications if we can find such a
            feature. We may be able to tune that feature up to make a model more
            likely to give "I don't know" responses rather than making something
            up. We may gain access to a method of identifying a hallucination
            that is agnostic to the length of a generation. The latter is a
            particularly exciting prospect because our current likelihood
            approach does not work well for long generations that contain
            several facts. The same is true of the semantic entropy technique we
            based our approach on, which requires longer generations to be
            decomposed into single facts before evaluation. But as I noted last
            time, finding a feature mapped to hallucinations, if it even exists,
            is finding a needle in a haystack. This search will be the subject
            of this blog.
        </p>
        <h3 class="h3">Design approach</h3>
        <p>
            The 3.8 billion parameter Phi-3-mini-4k-instruct model I'm using for
            this search has 101376 neurons across 33 layers, each activated to
            varying degrees during inference. This is tiny in LLM terms,
            effectively as small as a current generation LLM can be and still be
            coherent enough to be useful for everyday tasks. But even such a
            small model outputs a huge amount of data through its hidden states.
            I'm going to attack this problem with deep learning rather than a
            statistical approach because I'd rather make a computer do maths for
            me than do it myself.
        </p>
        <div class="items-center text-center">
            <i
                >Visualsation of hidden states for all 33 layers in
                Phi-3-mini-4k-instruct on a 36 token sequence</i
            >
        </div>
        <div
            class="flex justify-center mt-0"
            on:click={() => triggerImageModal(isometricStack)}
        >
            <img class="max-w-full h-auto" src={isometricStack} alt="map1" />
        </div>
        <div class="card variant-filled-ghost p-4">
            <Accordion>
                <AccordionItem closed>
                    <svelte:fragment slot="lead"
                        ><i class="fa-solid fa-code" /></svelte:fragment
                    >
                    <svelte:fragment slot="summary">Show code</svelte:fragment>
                    <svelte:fragment slot="content">
                        <p>
                            This image is an isometric view of the entire
                            model's activation states. I obtained the activation
                            map for each layer, for each token during generation
                            and created individual 2d maps for each layer. I
                            overlaid 2d maps to create the 3d visualisation.
                            Code to produce the 2d maps is available under the
                            next "Show code" dropdown below. Note that the code
                            for this blog is an update to the object we used in
                            the last blog and some global attributes and methods
                            are used. Code to generate the 3D maps is:
                        </p>
                        <CodeBlock
                            language="python"
                            code={`    def visualise_and_stack_layers(self, outputs, alpha=0.15, gap=100):
    """
    Method to visualise the activations of all layers and
    stack them into a single image.

    Args:
        outputs: dict: The outputs of the model.
        alpha: float: The alpha value for the image.
        gap: int: The gap between layers.

    Returns:
        bytes: The image bytes.
    """
    logging.info("LLMAgent visualising and stacking activations for sequence.")
    num_layers = len(outputs.hidden_states[0])
    logging.info("LLMAgent visualising activations for %s layers.", num_layers)
    layer_activation_images_bytes = []
    for i in range(num_layers):
        image_bytes = self.visualise_layer_activations(outputs, layer=i)
        layer_activation_images_bytes.append(image_bytes)
    logging.info("LLMAgent stacking activation images.")
    images = np.array(
        [
            Image.open(io.BytesIO(image_bytes)).resize((100, 100), Image.LANCZOS)
            for image_bytes in layer_activation_images_bytes
        ]
    )
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    num_images, height, width, _ = images.shape

    for i in range(num_images):
        img = images[i]
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = np.full_like(x, i * gap)

        img_normalized = img / 255.0
        facecolors = np.empty(img_normalized.shape, dtype=img_normalized.dtype)
        facecolors[..., :3] = img_normalized[..., :3]
        facecolors[..., 3] = img_normalized[..., 3] * alpha

        ax.plot_surface(
            x, y, z, rstride=1, cstride=1, facecolors=facecolors, shade=False
        )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, num_images * gap)
    ax.view_init(elev=30, azim=30)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_bytes = buffer.read()
    plt.close()

    return image_bytes`}
                        />
                    </svelte:fragment>
                </AccordionItem>
            </Accordion>
        </div>
        <p>
            To avoid shooting in the dark with so much data, I'm going to take
            some cues from Anthropic's approach in the
            <a
                href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html"
                ,
                class="text-blue-500 underline"
                >LLM feature extraction research</a
            >
            we discussed in the last blog. Anthropic used whole sequences, targeting
            their model's middle layer and using the residual stream rather than
            the Multi-Layer Perceptron (MLP) values. The residual stream is the value
            being passed between layers and altered based on attention and MLP values.
            The residual stream carries interlayer information and, happily, is the
            information returned to us by Transformers with
            <span class="pre p-1">output_hidden_states=True</span>. The next
            decision informing our approach is whether we can average neuron
            activation states through the entire generation or whether we need
            to break things down token by token. I visualised the data in order
            to feel out this point. The below image shows residual stream values
            for each token at layer 16 of the model during a response to a
            simple prompt, the tokens of which are overlaid.
        </p>
        <div
            class="flex justify-center mt-0"
            on:click={() => triggerImageModal(map1)}
        >
            <img class="max-w-full h-auto" src={map1} alt="map1" />
        </div>
        <div class="card variant-filled-ghost p-4">
            <Accordion>
                <AccordionItem closed>
                    <svelte:fragment slot="lead"
                        ><i class="fa-solid fa-code" /></svelte:fragment
                    >
                    <svelte:fragment slot="summary">Show code</svelte:fragment>
                    <svelte:fragment slot="content">
                        <p>
                            This image is a 2d visualisation of the resiudal
                            states at a given layer during a response. I
                            obtained the activation map for each token at the
                            layer and overlaid them to create the image. Code to
                            produce the 2d maps is:
                        </p>
                        <CodeBlock
                            language="python"
                            code={`    def visualise_layer_activations(self, outputs, layer=0):
    """
    Method to visualise the per-neuron activations for a given layer.

    Args:
        outputs: dict: The outputs of the model.
        layer: int: The layer to visualise.

    Returns:
        bytes: The image bytes.
    """
    logging.info(
        "LLMAgent visualising activations for layer %s for sequence.", layer
    )
    tokens = [
        self.processor.decode(input_token) for input_token in outputs.sequences[0]
    ]

    layer_feature_maps = []
    for tensor in outputs.hidden_states:
        target_layer = tensor[layer]
        tokens_in_tensor = target_layer.shape[1]
        for i in range(tokens_in_tensor):
            feature_map = target_layer[0, i, :].cpu().detach().numpy()
            layer_feature_maps.append(feature_map)

    total_tokens = len(layer_feature_maps)
    grid_size = int(np.ceil(np.sqrt(total_tokens)))
    plt.figure(figsize=(100, 100))
    plt.gca().patch.set_alpha(0)

    for idx, feature_map in enumerate(layer_feature_maps):
        n_activations = len(feature_map)
        heatmap_size = int(np.ceil(np.sqrt(n_activations)))
        padded_activations = np.pad(
            feature_map, (0, heatmap_size**2 - n_activations), mode="constant"
        )
        activation_grid = padded_activations.reshape(heatmap_size, heatmap_size)

        ax = plt.subplot(grid_size, grid_size, idx + 1)
        sns.heatmap(
            activation_grid,
            cmap="mako_r",
            cbar=False,
            linecolor="lightgrey",
            linewidths=0.2,
            xticklabels=False,
            yticklabels=False,
        )
        ax.text(
            0.5,
            0.5,
            tokens[idx],
            fontsize=80,
            color="white",
            ha="center",
            va="center",
            alpha=0.6,
            transform=ax.transAxes,
            weight="bold",
        )

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", transparent=True)
    buffer.seek(0)
    image_bytes = buffer.read()
    plt.close()
    logging.info("LLMAgent visualised activations for layer %s.", layer)
    return image_bytes`}
                        />
                    </svelte:fragment>
                </AccordionItem>
            </Accordion>
        </div>
        <p>
            It's a hard call here whether we can get away with averaging
            activations across these tokens or just picking a particular token's
            state and relying on the accumulation of meaning in the residual
            stream. We can see that there are some activation values that are
            distinctive throughout the entire generation, which suggests a
            strong degree of sequence-wide transferability (although there's a
            question here about how these activations are contributing meaning
            if their value is essentially fixed from the <span class="pre p-1"
                >&lt;|system|&gt;</span
            > token onwards). On the other hand, there seem to be some activation
            values that "flare" and then disappear back into the mileu, which could
            be lost through averaging across the sequence or choosing arbitrarily.
            I don't have a strong intuition about "where" the "lie" occurs on a token
            level when a LLM hallucinates.
        </p>
        <p>
            In doing some research to help make a reasonable choice on this
            point, I came across a paper from researchers at the Chinese Academy
            of Sciences titled
            <a
                href="https://arxiv.org/html/2312.16374v2"
                style="text-decoration: underline; color: lightblue;"
                >"LLM Factoscope: Uncovering LLMs’ Factual Discernment through
                Inner States Analysis"</a
            >. The paper investigates the line of inquiry we're on and strongly
            suggests that there is some merit to this idea. On my read of the
            paper, the researchers used the final layer activation map at the
            end of the prompt, without any response from the LLM. I think it's
            worth giving this a try with Anthropic's middle layer, whole
            sequence approach as well, so will run four experiments to see how
            things shake out. The experiments will attempt to identify a feature
            that is highly correlated with hallucinations based on:
        </p>
        <div class="grid grid-cols-2 gap-4">
            <div
                class="flex justify-center mt-0"
                on:click={() => triggerImageModal(map2)}
            >
                <div class="items-center text-center">
                    <i>Layer 16, end of prompt</i>
                    <img
                        class="max-h-[400px] max-w-[400px] max-w-full h-auto"
                        src={map2}
                        alt="map1"
                    />
                </div>
            </div>
            <div
                class="flex justify-center mt-0"
                on:click={() => triggerImageModal(map3)}
            >
                <div class="items-center text-center">
                    <i>Layer 16, end of response</i>
                    <img
                        class="max-h-[400px] max-w-[400px] max-w-full h-auto"
                        src={map3}
                        alt="map1"
                    />
                </div>
            </div>
            <div
                class="flex justify-center mt-0"
                on:click={() => triggerImageModal(map4)}
            >
                <div class="items-center text-center">
                    <i>Layer 33, end of prompt</i>
                    <img
                        class="max-h-[400px] max-w-[400px] max-w-full h-auto"
                        src={map4}
                        alt="map1"
                    />
                </div>
            </div>
            <div
                class="flex justify-center mt-0"
                on:click={() => triggerImageModal(map5)}
            >
                <div class="items-center text-center">
                    <i>Layer 33, end of response</i>
                    <img
                        class="max-h-[400px] max-w-[400px] max-w-full h-auto"
                        src={map5}
                        alt="map1"
                    />
                </div>
            </div>
        </div>
        <div class="card variant-filled-ghost p-4">
            <Accordion>
                <AccordionItem closed>
                    <svelte:fragment slot="lead"
                        ><i class="fa-solid fa-code" /></svelte:fragment
                    >
                    <svelte:fragment slot="summary">Show code</svelte:fragment>
                    <svelte:fragment slot="content">
                        <p>
                            This image is a 2d visualisation of the resiudal
                            states at a given layer and token during a response.
                            I obtained the activation map for each token at the
                            layer and overlaid them to create the image. Note
                            that we also have an optional parameter to return
                            the raw numeric state for the layer and token, which
                            is used in dataset generation.
                        </p>
                        <CodeBlock
                            language="python"
                            code={`    def visualise_activation_map_at_layer_at_token(
                                self, outputs, layer, token, return_numeric_state=True
                            ):
    """
    Method to visualise the activation map at a given layer
    and token.

    Args:
        outputs: dict: The outputs of the model.
        layer: int: The layer to visualise.
        token: int: The token to visualise.
        return_numeric_state: bool: Whether to return the raw state instead of image bytes.

    Returns:
        bytes: The image bytes. | numpy.ndarray: The numeric activation state.
    """
    logging.info(
        "LLMAgent visualising activations for layer %s and token %s.",
        layer,
        token,
    )
    tokens = [
        self.processor.decode(input_token) for input_token in outputs.sequences[0]
    ]

    layer_feature_maps = []
    for tensor in outputs.hidden_states:
        target_layer = tensor[layer]
        tokens_in_tensor = target_layer.shape[1]
        for i in range(tokens_in_tensor):
            feature_map = target_layer[0, i, :].cpu().detach().numpy()
            layer_feature_maps.append(feature_map)

    token_feature_map = layer_feature_maps[token]
    if return_numeric_state:
        return token_feature_map

    total_tokens = len(layer_feature_maps)
    grid_size = int(np.ceil(np.sqrt(total_tokens)))
    plt.figure(figsize=(10, 10))
    plt.gca().patch.set_alpha(0)

    n_activations = len(token_feature_map)
    heatmap_size = int(np.ceil(np.sqrt(n_activations)))
    padded_activations = np.pad(
        token_feature_map, (0, heatmap_size**2 - n_activations), mode="constant"
    )
    activation_grid = padded_activations.reshape(heatmap_size, heatmap_size)
    ax = plt.subplot(1, 1, 1)
    sns.heatmap(
        activation_grid,
        cmap="mako_r",
        cbar=False,
        linecolor="lightgrey",
        linewidths=0.2,
        xticklabels=False,
        yticklabels=False,
    )
    ax.text(
        0.5,
        0.5,
        tokens[token],
        fontsize=80,
        color="white",
        ha="center",
        va="center",
        alpha=0.6,
        transform=ax.transAxes,
        weight="bold",
    )

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", transparent=True)
    buffer.seek(0)
    image_bytes = buffer.read()
    plt.close()
    logging.info(
        "LLMAgent visualised activations for layer %s and token %s.", layer, token
    )
    return image_bytes`}
                        />
                    </svelte:fragment>
                </AccordionItem>
            </Accordion>
        </div>

        <h3 class="h3">Implementation</h3>

        <p>
            To run our experiments we're going to need a lot of tagged
            hallucination/non-hallucination data. I'm am going to stick to the
            TriviaQA dataset. We want to collect at least a few thousand data
            points where we have the following:
        </p>

        <ul>
            <li>- TriviaQA question</li>
            <li>- TriviaQA answer</li>
            <li>- LLM response</li>
            <li>- LLM response correctness</li>
            <li>- Middle layer activation state for prompt</li>
            <li>- Final layer activation state for prompt</li>
            <li>- Middle layer activation state for response</li>
            <li>- Final layer activation state for response</li>
        </ul>

        <p>
            We can make the evaluation of answer correctness unattended by using
            sentence embeddings to evaluate the semantic closeness between the
            LLM's output and the canonical answer. I picked a high-performing
            small model from the <a
                href="https://huggingface.co/spaces/mteb/leaderboard"
                style="text-decoration: underline; color: lightblue;"
                >MTEB embedding benchmark leaderboard</a
            >, MixedBread AI's
            <a
                href="https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1"
                style="text-decoration: underline; color: lightblue;"
                >mxbai-embed-large-v1</a
            >
            . We use the model to produce vectors for dataset answers and model outputs
            and then evaluate their cosine similarity, with closer distances (closer
            to 1.0) representing more semantic alignment. I ran off a test dataset
            and decided to use 0.68 as the similarity threshold to regard two answers
            as the same.
        </p>
        <div class="card variant-filled-ghost p-4">
            <Accordion>
                <AccordionItem closed>
                    <svelte:fragment slot="lead"
                        ><i class="fa-solid fa-code" /></svelte:fragment
                    >
                    <svelte:fragment slot="summary">Show code</svelte:fragment>
                    <svelte:fragment slot="content">
                        <p>
                            I used the function below to determine whether model
                            outputs were close matches to answers.
                        </p>
                        <CodeBlock
                            language="python"
                            code={`    def check_answer(self, query, comparisons):
    """
    Method to check is an output answer is close
    enough to any provided comparison answers.

    Args:
        query: str: The first sentence.
        comparisons: list[str]: Comparison sentences.

    Returns:
        bool: True if the sentence matches any of the comparisons.
        float: The max similarity score.
    """
    if not query.startswith(self.embedding_instruction):
        query = self.embedding_instruction + query
    inputs = [query] + comparisons
    embeddings = self.embedding_model.encode(inputs)
    similarities = cos_sim(embeddings[0], embeddings[1:])[0].tolist()
    is_same = False
    for similarity in similarities:
        if similarity > self.embedding_sameness_threshold:
            return True, max(similarities)
    return False, max(similarities)`}
                        />
                    </svelte:fragment>
                </AccordionItem>
            </Accordion>
        </div>
        <p>
            I also collected information on the more naive output
            probability-based metrics we discussed in the last blog so we can
            evaluate the performance of whatever features we find against
            metrics we know are effective. The last decision is whether we want
            to store the activation maps as image data like we've been
            displaying or to directly operate on the the floating point values.
            I'm going to operate directly on the values, but generally
            converting non-image signals (for example, sound) to images is a
            reasonable approach to this type of deep learning problem. I created
            a method to stream the dataset from Huggingface, perform out
            processing and stream results to a ndJSON file on disk.
        </p>
        <div class="card variant-filled-ghost p-4">
            <Accordion>
                <AccordionItem closed>
                    <svelte:fragment slot="lead"
                        ><i class="fa-solid fa-code" /></svelte:fragment
                    >
                    <svelte:fragment slot="summary">Show code</svelte:fragment>
                    <svelte:fragment slot="content">
                        <CodeBlock
                            language="python"
                            code={`    def test_on_triviaQA(self, filename="triviaQA.ndjson", n=100):
    """
    Method to test hallucination detection methods on the
    TriviaQA dataset. Streams results into an ndjson file.

    Args:
        filename: str: The filename to save the results to.
        n: int: The number of samples to test on.
    """
    logging.info(
        "LLMAgent generating hallucination detection dataset from TriviaQA with %s samples.",
        n,
    )
    dataset = load_dataset("trivia_qa", "rc", split="train", streaming=True)
    iterator = iter(dataset)
    for i in range(n):
        logging.info("LLMAgent processing TriviaQA sample %s of %s.", i + 1, n)
        entry = next(iterator)
        question = entry["question"]
        answer = entry["answer"]["value"]
        (
            response,
            total_probability,
            average_token_probability,
            _,
            generate_output,
        ) = self.generate_with_probability(question, max_tokens=15)
        is_same, max_similarity = self.check_answer(response, [answer])
        target_token = self.processor.encode(self.assistant_prompt_start)[0]
        for i, token_id in enumerate(generate_output.sequences[0]):
            if token_id == target_token:
                target_token_index = i - 1
                break
        row = {
            "question": question,
            "answer": answer,
            "response": response,
            "correct": is_same,
            "similarity": max_similarity,
            "total_probability": total_probability,
            "total_probability_predicts": (
                True if total_probability > 0.5 else False
            ),
            "average_token_probability": average_token_probability,
            "average_token_propability_predicts": (
                True if average_token_probability > 0.75 else False
            ),
            "both_metrics_predict": (
                True
                if (total_probability + average_token_probability) / 2 > 0.625
                else False
            ),
            "middle_layer_activations_prompt": self.visualise_activation_map_at_layer_at_token(
                generate_output, 16, target_token_index, return_numeric_state=True
            ).tolist(),
            "middle_layer_activations_response": self.visualise_activation_map_at_layer_at_token(
                generate_output, 16, -1, return_numeric_state=True
            ).tolist(),
            "final_layer_activations_prompt": self.visualise_activation_map_at_layer_at_token(
                generate_output, -1, target_token_index, return_numeric_state=True
            ).tolist(),
            "final_layer_activations_response": self.visualise_activation_map_at_layer_at_token(
                generate_output, -1, -1, return_numeric_state=True
            ).tolist(),
        }
        with open(filename, "a") as file:
            file.write(json.dumps(row) + "\\n")`}
                        ></CodeBlock>
                    </svelte:fragment>
                </AccordionItem>
            </Accordion>
        </div>
        <h3 class="h3">Findings</h3>
        <p>
            I generated a dataset of 10,000 rows and trained a few simple neural
            network architectures as binary classifiers. I was able to get back
            up to the state-of-the-art range, topping out at 79.80% accuracy
            using a deep feedforward neural network with five fully connected
            layers and ReLU activations, targeting the model's middle layer
            activation states for the response. Across all classifier models,
            evaluating the response activation map performed better than
            evaluating the prompt activation map and evaluating the middle layer
            activation map performed better than evaluating the final layer.
            Noting this, performance differences between the best and worst
            classifier architecture and approach were around 5%.
        </p>
        <div
            class="container h-full mx-auto flex justify-center items-center leading-relaxed"
        >
            <div class="space-y-2 m-3 custom-container">
                <div class="table-container">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Classification model</th>
                                <th colspan="2" class="text-center">Middle</th>
                                <th colspan="2" class="text-center">Final</th>
                            </tr>
                            <tr>
                                <th></th>
                                <th>Prompt end</th>
                                <th>Response end</th>
                                <th>Prompt end</th>
                                <th>Response end</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>3 layer feedforward, ReLU</td>
                                <td class="text-center">0.7580</td>
                                <td class="text-center">0.7850</td>
                                <td class="text-center">0.7475</td>
                                <td class="text-center">0.7785</td>
                            </tr>
                            <tr>
                                <td
                                    >5 layer feedforward, Leaky ReLU, BN+dropout</td
                                >
                                <td class="text-center">0.7525</td>
                                <td class="text-center">0.7890</td>
                                <td class="text-center">0.7470</td>
                                <td class="text-center">0.7785</td>
                            </tr>
                            <tr>
                                <td>5 layer feedforward, ReLU</td>
                                <td class="text-center">0.7490</td>
                                <td class="text-center text-teal-600">0.7980</td
                                >
                                <td class="text-center text-red-500">0.7440</td>
                                <td class="text-center">0.7780</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <p>
            The next question to evaluate is whether the classifier always
            agrees with the probability-based predictions or whether there is
            some divergence. I reviewed every item in the dataset using the best
            performing model and comparing the prediction accuracy with the
            prediction based on the combined likelihood metric I outlined in the
            last blog post. The classifiers disagree in 22% of instances, which
            suggests a combined metric might push performance even higher, given
            the approaches are demonstrating different strengths. In instances
            where the classifiers disagreed, the model was right 58% of the time
            and the likelihood metric 42% of the time. False positive and false
            negative ratios for the classifier and naivie likelihood metric show
            a better balance for the classifier model and a strong skew towards
            erroneous hallucination identification for the likelihood metric.
        </p>
        <div
            class="container h-full mx-auto flex justify-center items-center leading-relaxed"
        >
            <div class="space-y-2 m-3 custom-container">
                <div class="table-container">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Approach</th>
                                <th>False positive</th>
                                <th>False negative</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Naive likelihood</td>
                                <td class="text-center">0.69</td>
                                <td class="text-center">0.31</td>
                            </tr>
                            <tr>
                                <td>Classifier model</td>
                                <td class="text-center">0.53</td>
                                <td class="text-center">0.47</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <p>
            There is probably some further insight to be gained by examining the
            failure cases, but I'm satisfied that we've proved the existence of
            a discernable hallucination feature in hidden states. But just what
            does the hallucination feature look like? I created a saliency map
            from the binary classification model and remapped it over the same
            neuron-heatmap we've been using to visualise activations so far.
            This saliency map shows how heavily the binary classifier we trained
            weights each of the activation states in the model when it is
            determining whether an output is likely to be hallucinatory or
            factual (lighter means more impact on the output). We don't know
            whether all of these features correlate positively with
            hallucinations (some might strongly track with a truthy prediction)
            but we can be sure that the hallucination features that our
            classifier is responding to are present on this graph.
            Interestingly, most neurons appear to have no bearing on this
            determination and few neurons are highly weighted, suggesting the
            possibility of a relatively straightforward and well-defined
            hallucination feature.
        </p>
        <div
            class="flex justify-center mt-0"
            on:click={() => triggerImageModal(map6)}
        >
            <div class="items-center text-center">
                <img class="max-w-full h-auto" src={map6} alt="map6" />
            </div>
        </div>
        <p>
            I want to leave off with a note that if your objective is increasing
            an LLM's tendency to admit that it does not know something rather
            than hallucinating out a guess, a much simpler approach than the
            above might be to use a <a
                href="https://vgel.me/posts/representation-engineering/"
                style="text-decoration: underline; color: lightblue;"
                >control vector</a
            >
            . Control vectors are a bias added to each of an LLM's layers, generated
            automatically by making the model act a certain way with prompts and
            then performing principal component analysis to find what makes those
            outputs distinctive. It's quite a similar idea to what we looked into
            here, but likely to be much more expedient, because you don't need to
            find the needle in a haystack feature, you just start throwing lots of
            proximate data in and see what sticks.
        </p>
    </div>
</div>

<style>
    @media (max-width: 639px) {
        .custom-container {
            max-width: calc(100% - 2rem);
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    @media (min-width: 640px) and (max-width: 767px) {
        .custom-container {
            max-width: calc(100% - 4rem);
            padding-left: 2rem;
            padding-right: 2rem;
        }
    }
    @media (min-width: 768px) and (max-width: 1023px) {
        .custom-container {
            max-width: calc(100% - 6rem);
            padding-left: 3rem;
            padding-right: 3rem;
        }
    }
    @media (min-width: 1024px) {
        .custom-container {
            max-width: calc(100% - 8rem);
            padding-left: 4rem;
            padding-right: 4rem;
        }
    }
</style>
