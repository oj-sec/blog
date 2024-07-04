<script lang="ts">
    import { getModalStore } from "@skeletonlabs/skeleton";
    const modalStore = getModalStore();

    const cImage =
        "max-w-[90%] max-h-[90%] rounded-container-token overflow-hidden shadow-xl";
</script>

{#if $modalStore[0]}
    <img
        src={$modalStore[0]?.image}
        class={cImage}
        alt="Example"
        title="Source: {$modalStore[0]?.meta.source}"
    />
{/if}
