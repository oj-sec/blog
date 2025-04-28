<script lang="ts">
    import { getModalStore } from "@skeletonlabs/skeleton";
    const modalStore = getModalStore();

    const cImage = "max-w-[85rem] max-h-[48rem] w-[90%] h-[90%] rounded-container-token overflow-hidden shadow-xl";
</script>

{#if $modalStore[0]}
    <div class="flex items-center justify-center overflow-auto">
        <img
            src={$modalStore[0]?.image}
            class={cImage}
            alt="fullscreen"
            title="Source: {$modalStore[0]?.meta.source}"
        />
    </div>
{/if}