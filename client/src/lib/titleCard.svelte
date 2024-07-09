<script>
    export let title;
    export let date;
    export let href;
    export let timeToRead;
    export let tags = [];

    import TagList from "$lib/tagList.svelte";

    function navigate(url) {
        window.location.href = url;
    }
</script>

<div
    class="card variant-filled-surface p-4 my-6 card-hover"
    on:click={() => navigate(href)}
>
    <header class="card-header">
        <h3 class="text-xl">
            {title}
        </h3>
    </header>
    <footer class="card-footer mt-2">{date} - {timeToRead}</footer>
    <div class="ml-4 mt-0">
        <TagList {tags} />
    </div>
</div>
