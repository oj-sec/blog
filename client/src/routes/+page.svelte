<script>
    import TitleCard from "$lib/titleCard.svelte";

    function navigate(url) {
        window.location.href = url;
    }

    let posts = [
        {
            title: "Hot Leads: Large Scale Phishing Against Marketing Sector Deploys Novel Python Infostealer",
            date: "3 December 2024",
            href: "/blog/20241203",
            timeToRead: "13 min read",
            tags: [
                "threat intelligence",
                "malware analysis",
                "information stealer",
                "ecrime",
                "distribution",
                "initial access",
            ],
        },
        {
            title: "Research aside - Hallucination detection & LLM explainability - Part 2",
            date: "9 July 2024",
            href: "/blog/20240709",
            timeToRead: "11 min read",
            tags: [
                "ai",
                "technical",
                "deep-dive",
                "advanced",
                "research",
                "llm",
                "hallucination",
                "explainability",
            ],
        },
        {
            title: "Research aside - Hallucination detection & LLM explainability - Part 1",
            date: "5 July 2024",
            href: "/blog/20240705",
            timeToRead: "12 min read",
            tags: [
                "ai",
                "technical",
                "deep-dive",
                "advanced",
                "research",
                "llm",
                "hallucination",
                "explainability",
            ],
        },
        {
            title: "Evaluating Large Language Models as future event forecasters - Part Two: Performance & token sampling",
            date: "5 May 2024",
            href: "/blog/20240505",
            timeToRead: "8 min read",
            tags: [
                "ai",
                "intelligence analysis",
                "forecasting",
                "introductory",
                "technical",
            ],
        },
        {
            title: "Evaluating Large Language Models as future event forecasters - Part One: Introduction",
            date: "4 April 2024",
            href: "/blog/20240404",
            timeToRead: "8 min read",
            tags: [
                "ai",
                "intelligence analysis",
                "forecasting",
                "introductory",
                "technical",
            ],
        },
    ];
</script>

<div class="container h-full mx-auto flex justify-center items-center">
    <div class="space-y-5 m-10">
        <h1 class="h1 text-center mb-0" on:click={() => navigate("/")}>
            @oj-sec
        </h1>
        <p class="text-sm text-center mb-20">
            <img /><a class="no-underline hover:underline" href="/blog/about"
                >Oliver Smith
            </a><br />
            <span
                >security | intelligence analysis | artificial intelligence</span
            >
        </p>
        {#each posts as post}
            <TitleCard {...post} />
        {/each}
    </div>
</div>
