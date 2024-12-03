<script>
    export let attackData;
</script>

<!-- Responsive Container (recommended) -->
<div class="table-container">
    <!-- Native Table Element -->
    <table class="table table-hover">
        <thead>
            <tr>
                <th>Tactic</th>
                <th>Technique</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            {#each Object.keys(attackData) as key}
                {#if attackData[key].length > 0}
                    {#each attackData[key] as technique, index}
                        <tr>
                            {#if index === 0}
                                <td rowspan={attackData[key].length}>{key}</td>
                            {/if}
                            <td>{technique.technique}</td>
                            <td>{technique.description}</td>
                        </tr>
                    {/each}
                {/if}
            {/each}
        </tbody>
    </table>
</div>

