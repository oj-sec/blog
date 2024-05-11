<script>
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

        <h2 class="h2">About me</h2>
        <p>
            I'm Oliver Smith, an Australian intelligence analyst and information
            security professional with a background in law enforcement. On this
            blog, I'll be writing about threat intelligence, artificial
            intelligence and cyber investigations.
        </p>
        <p>
            I work as a threat intelligence lead at an Australian cyber security
            Managed Service provider. I hold teriary qualifications in law,
            international relations, police studies and computer science. I am
            also a designated detective and hold certifications including a GIAC
            GCTI. If you have any questions about the content of this blog or
            would otherwise like to get in touch, you can reach me at <span
                class="underline">oliver@oj-sec.com</span
            >.
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
