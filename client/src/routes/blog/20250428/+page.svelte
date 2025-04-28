<script>
    import { getModalStore } from "@skeletonlabs/skeleton";
    import TagList from "$lib/tagList.svelte";

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
        "threat intelligence",
        "data analysis",
        "tool",
        "text embedding",
    ];

    // Image carousel
    let selectedImageIndex = 0;
    $: isFinalImage = selectedImageIndex === images.length - 1;
    $: isFirstImage = selectedImageIndex === 0;

    function nextImage() {
        if (selectedImageIndex >= images.length - 1) {
            return;
        }
        selectedImageIndex = selectedImageIndex + 1;
    }
    function previousImage() {
        if (selectedImageIndex <= 0) {
            return;
        }
        selectedImageIndex = selectedImageIndex - 1;
    }
    import splash from "./splash.png";
    import image0 from "./0.png";
    import image1 from "./1.png";
    import image2 from "./2.png";
    import image3 from "./3.png";
    import image4 from "./4.png";
    import image5 from "./5.png";
    import image6 from "./6.png";
    import image7 from "./7.png";
    import image8 from "./8.png";
    import image9 from "./9.png";
    import image10 from "./10.png";
    import image11 from "./11.png";
    import image12 from "./12.png";
    import image13 from "./13.png";

    let images = [
        {
            src: image0,
            text: "We start by loading our JSON data into the Shadowpuppet UI to create the working database.",
        },
        {
            src: image1,
            text: "After previewing the data, we configure our embeddings and target the message field.",
        },
        {
            src: image2,
            text: "And set our embeddings running.",
        },
        {
            src: image3,
            text: "After they're finished, we can choose to visualise the newly created database.",
        },
        {
            src: image4,
            text: "And then configure our projection, which can currently handle the PaCMAP projection technique.",
        },
        {
            src: image5,
            text: "We then have our visualisation, and can select points to view the underlying data.",
        },
        {
            src: image6,
            text: 'We can find points containing the substrings "ransom" and "negotiat*" and highlight them red and green respectively.',
        },
        {
            src: image7,
            text: 'Hits for "ransom" consist of Black Basta operators discussing english-language media coverage of their operations and hits for "negotiat*" highlight apparent instances of operators clearing messages to victims with management before sending them.',
        },
        {
            src: image8,
            text: "We can alternatively highlight based on timestamps by creating a sequential rule with 12 buckets to cover the 12 month period contained in the leaks.",
        },
        {
            src: image9,
            text: "This results in a visualisation where the colour of the points get darker as as time goes on.",
        },
        {
            src: image10,
            text: "We can pick up a semantically and temporally aligned message cluster that consists of an exchange between operators about DLL malware delivery formats and AV evasion.",
        },
        {
            src: image11,
            text: "Finally, we can use a category rule to highlight points based on the sender.",
        },
        {
            src: image12,
            text: "This creates a visualisation where each operator has a unique colour, with some distinctive clusters.",
        },
        {
            src: image13,
            text: 'A distinctive cluster of messages are from an apparent junior Black Basta affiliate with the handle "arslanshabbirmalik" who communicates in emoji-rich, deferential english.',
        },
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
            Shadowpuppet: Analysing Unstructured Data Using Semantic Scatter
            Plots
        </h2>
        <p>28 April 2025 - 10 min read</p>
        <TagList {tags} />
        <p class="card variant-filled-ghost p-4">
            You can find source code and executables for Shadowpuppet, the tool
            discussed in this blog, <a
                href="https://github.com/oj-sec/shadowpuppet"
                style="text-decoration: underline; color: lightblue;">here</a
            >.
        </p>
        <div class="flex justify-center mt-0 flex-1 relative max-w-4/5">
            <img class="max-w-full h-auto" src={splash} alt="splash" />
        </div>
        <h3 class="h3">Introduction</h3>
        <p>
            Semantic scatter plots are graphs that represent points with a
            closer meaning more closely on a graph. This method of representing
            data can uncover patterns that are difficult to discern using
            conventional approaches, particularly over very large natural
            language and code datasets. Semantic scatter plots provide a deeply
            contextual view into dataset because it's the relative nuance within
            the data being represented rather than objective features. This can
            make patterns that we wouldn't know to look for self-revealing.
        </p>
        <p>
            In this blog, I'll introduce a semantic scatter plot tool I've been
            working on called Shadowpuppet. I'll provide some background on
            semantic scatter plots and a quick demo of the tool on
            communications leaked from the Black Basta cyber extortion group.
            This blog should set you up with some ideas to run this technique on
            your own data.
        </p>
        <h3 class="h3">Semantic Scatter Plots</h3>
        <p>
            The first step in creating a semantic scatter plot is to use an
            embedding model to create a high-dimensional representation of the
            meaning of source data. Embedding models are similar to the encoder
            portion of a Large Language Models (LLMs) in that both serve to
            represent input data numerically in a latent space based on patterns
            learned through the magic of training transformers over a massive
            dataset. Embedding models differ in that their entire purpose is to
            output these numeric representations, called vectors, with no
            decoder or generative stage. The output of an embedding model is
            specialised for creating representations that capture similarity and
            contrast.
        </p>
        <p>
            The concept of the high-dimensional representation is important for
            us to understand. Embedding models output vectors containing long
            sequences, commonly 768 or 1024 numbers in an array. These numbers
            are just like <span class="pre p-1 break-all">[x,y]</span>
            coordinates in a 2d space or
            <span class="pre p-1 break-all">[x,y,z]</span> coordinates in a 3d space,
            only in a space that is massive, hugely beyond our ability to think reason
            about in real world spatial terms. The complexity of the space scales
            exponentially with each added dimension, giving us a hugely descriptive
            space in which our data's meaning can be chraracterised.
        </p>
        <p>
            Once we have our high-dimensional vectors, our next step is to use a
            dimension reduction technique to get back down into a 2d space we
            can graph. By way of example, dimension reduction is like making a
            shadow puppet using light projected against a wall. We take a
            three-dimensional object, our hand, and project a two-dimensional
            shape onto the wall. The projection still contains a lot of
            important spatial data that evokes the original shape and gives us a
            simpler representation we can more easily understand.
        </p>
        <p>
            Approaches to achieving dimension reduction generally revolve around
            some form of principal component analysis where important
            high-dimensional structure is preserved as we project down. It's the
            combination of these two phases that make this a powerful technique
            - we get an incredibly rich representation of the data initially
            using embeddings and then get a deeply contextual final product when
            we reduce dimensions.
        </p>
        <h3 class="h3">Introducing Shadowpuppet</h3>
        <p>
            Semantic scatter plots are a tool I've found myself throwing at a
            lot of threat intelligence problems over the last year. I've put
            together Shadowpuppet to streamline my workflow and hope it might be
            useful to share with others as well. Examples of instances where
            semantic scatter plots have let me perform unique analysis in a
            threat intelligence context have included:
        </p>
        <ul class="list-disc ml-4">
            <li>
                analysing more than 50,000 social media posts produced by an
                information operation capability to identify coordination in
                political narratives
            </li>
            <li>
                analysing Chrome browser API invocation in more than 100,000
                browser extensions to pivot from known suspicious activity to
                behaviorally similar extensions
            </li>
        </ul>
        <p>
            Shadowpuppet is available for Windows, macOS and Linux and uses
            embeddings computed locally, making it potentially suitable for
            handling sensitive data. For Windows and macOS, I've produced
            bundled Tauri executables that mean you don't need to open your
            terminal to run Shadowpuppet. Shadowpuppet currently supports
            Sentence Transformer-compatible embedding models for embeddings and
            the PaCMAP technique for dimension reduction.
        </p>
        <p>
            Shadowpuppet works by reading input data into a local sqlite
            database and embedding a target field. Embeddings are very
            computationally expensive, so Shadowpuppet writes embeddings into
            the database to avoid double processing. Each time a database is
            visualised, a new projection is created and the user can view point
            data by clicking on points. The user can also create complex rules
            to highlight certain points based on queries executed against the
            database. This process of dynamically exploring the data is the key
            feature of Shadowpuppet. Shadowpuppet should comfortably handle
            datasets well into the hundreds of thousands of rows on most
            devices.
        </p>
        <h3 class="h3">Demo on the Black Basta Leaks</h3>
        <p>
            I'm going to briefly demo Shadowpuppet using matrix chat logs leaked
            from the Black Basta cyber extortion group in February 2025. My
            intention is not to provide a comprehensive analysis of the leaks,
            rather how you'd go about loading and starting to explore this data
            using Shadowpuppet. The Black Basta leak was originally publicly
            identified by <a
                href="https://x.com/PRODAFT/status/1892636346885235092"
                style="text-decoration: underline; color: lightblue;">PRODAFT</a
            >
            and I sourced the data used here from
            <a
                href="https://github.com/D4RK-R4BB1T/BlackBasta-Chats"
                style="text-decoration: underline; color: lightblue;"
                >D4RK-R4BB1T</a
            >.
        </p>
        <p>
            I started by cleaning up the data with a script to reformat the
            messages into valid JSON then fired up Shadowpuppet. The leak
            consists of 195,882 messages, primarily in Russian, but including
            some mixed-language content and some code. You may wish to translate
            the data before you run an analysis like this, but it's worth noting
            that many embedding models are multilingual and will handle
            mixed-language data. The leaks include a timestamp, a message sender
            and message text. We're going to target the message text as the
            field we embed but I'll demonstrate how we can use the other fields
            in our analysis. Below is a walkthrough of using Shadowpuppet to
            explore the Black Basta leaks.
        </p>
        <div class="card variant-surface pt-4 pb-2">
            {#if images.length > 0}
                <div class="mx-auto flex justify-center items-center w-full">
                    <div class="max-w-5/6 w-5/6 text-center">
                        <i>{images[selectedImageIndex].text}</i>
                    </div>
                </div>
                <div
                    class="w-full flex justify-between items-stretch mx-auto my-2"
                >
                    <button
                        type="button"
                        class="btn rounded-sm variant-surface px-8 self-stretch ml-2 flex items-center justify-center {isFirstImage
                            ? 'invisible'
                            : ''}"
                        on:click={previousImage}
                    >
                        <i class="fa-solid fa-chevron-left"></i>
                    </button>
                    <div class="flex justify-center mt-0 flex-1 relative">
                        <img
                            class="max-w-full h-auto"
                            src={images[selectedImageIndex].src}
                            alt={images[selectedImageIndex].alt ||
                                "gallery image"}
                        />
                    </div>
                    <button
                        type="button"
                        class="btn rounded-sm variant-surface px-8 self-stretch ml-2 flex items-center justify-center {isFinalImage
                            ? 'invisible'
                            : ''}"
                        on:click={nextImage}
                    >
                        <i class="fa-solid fa-chevron-right"></i>
                    </button>
                </div>
            {/if}
        </div>
        <p>
            I'll keep adding features to Shadowpuppet as my workflow evolves but
            if there's something that would be useful to you, please let me know
            in <a
                href="https://github.com/oj-sec/shadowpuppet"
                style="text-decoration: underline; color: lightblue;"
                >an issue on the repository</a
            >. Additional features are likely to include the ability to embed
            labels, preset high-contrast colour scales to increase visual
            clarity and handlers for more embedding models and dimension
            reduction techniques.
        </p>
        <p>
            Whether you're sifting through communications data, analysing code
            at scale, hunting over command line history or tracking information
            operations, semantic scatter plots are a useful technique to have in
            your toolkit. They can help you find patterns that you might not
            have thought to look for and can be a great first step on large
            datasets. Hopefully this blog has given you some ideas for how
            semantic scatter plots could be useful to you and a running start to
            get visualising. Happy hunting!
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
