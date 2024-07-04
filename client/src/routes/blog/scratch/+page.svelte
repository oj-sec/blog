<script>
    import { data } from "./arenaData.js";
</script>

<div
    class="container h-full mx-auto flex justify-center items-center leading-relaxed"
>
    <div class="space-y-5 m-10 custom-container">
        <!-- Responsive Container (recommended) -->
        <div class="table-container">
            <!-- Native Table Element -->
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Elo</th>
                        <th>Model</th>
                        <th>Organisation</th>
                        <th>Weights</th>
                        <th>Country</th>
                    </tr>
                </thead>
                <tbody>
                    {#each data.slice(10, 20) as row, i}
                        <tr>
                            <td>{row.rank}</td>
                            <td>{row.elo}</td>
                            <td>{row.model}</td>
                            <td>{row.organisation}</td>
                            <td
                                ><span
                                    class="chip variant-filled-{row.weights ===
                                    'Open'
                                        ? 'success'
                                        : 'error'}">{row.weights}</span
                                ></td
                            >
                            <td class="flex justify-center items-center"
                                ><span class="text-center items text-2xl"
                                    >{row.country === "US"
                                        ? "🇺🇸"
                                        : row.country === "China"
                                          ? "🇨🇳"
                                          : row.country === "France"
                                            ? "🇫🇷"
                                            : row.country}</span
                                ></td
                            >
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
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
