<script>
    export let tags = [];
</script>

{#if tags.length > 0}
    <div class="flex flex-wrap items-left justify-left">
        {#each tags as tag}
            <span class="chip variant-filled mr-2 my-2">{tag}</span>
        {/each}
    </div>
{/if}
