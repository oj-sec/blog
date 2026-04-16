<script>
    import TagList from "$lib/tagList.svelte";
    function navigate(url) {
        window.location.href = url;
    }

    let tags = ["intelligence analysis", "visualisation", "forecasting"];

    import rehoboam from "./rehoboam.gif";
</script>

<div
    class="container h-full mx-auto flex justify-center items-center leading-relaxed"
>
    <div class="space-y-5 m-10 custom-container">
        <h1 class="h1 text-center mb-12" on:click={() => navigate("/")}>
            @oj-sec
        </h1>
        <h2 class="h2">Polymarket as a Forecasting Tool</h2>
        <p>11 April 2026 - 8 min read</p>
        <TagList {tags} />
        <!-- <div class="flex justify-center mt-0 flex-1 relative">
            <img class="max-h-[600px]" src={splash} alt="splash" />
        </div> -->
        <h3 class="h3">Background</h3>
        <p>
            Prediction markets are a well-studied tool for forecasting future
            events. The basic concept is that market participants can place bets
            on the outcome of future events and that price discovery through bet
            supply and demand sharpens market consensus towards well-reasoned
            predictions. <a
                href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4691513"
                style="text-decoration: underline; color: lightblue;"
                >Academic research</a
            >
            indicates that naive prediction markets lag small expert team forecasting
            accuracy, but still perform significantly better than chance.
        </p>
        <p>
            Prediction markets are also much more accessible to outside
            observers than small elite teams. Polymarket is a prediction market
            that has operated since 2020. Polymarket functions via
            cryptocurrency tokens that represent shares in the outcome of a
            future event and exposes extensive market detail through its APIs.
            Polymarket is not a pure prediction market. It is axiomatic that
            there is an information asymmetry between market participants and
            there is <a
                href="https://www.theguardian.com/business/2026/apr/08/polymarket-trump-us-iran-ceasefire"
                style="text-decoration: underline; color: lightblue;"
                >clear evidence</a
            >
            that insiders have used Polymarket to place bets using privileged knowledge.
            The extent to which this market distortion reduces Polymarket's viability
            as an economic marketplace is an open question. Polymarket's token-based
            consensus system may also create situations where markets are erroneously
            resolved with
            <a
                href="https://polymarket.com/event/major-cyberattack-on-iran-in-june"
                style="text-decoration: underline; color: lightblue;"
                >an outcome arguably contrary to fact</a
            >. Nonetheless, it is an interesting data source for exploring the
            wisdom (or not) of the crowd.
        </p>
        <h3 class="h3">Polymarket Data</h3>
        <p>
            I used Polymarket APIs to obtain all closed markets since 1 January
            2025 where there was a minimum trading volume of $2000 and where the
            market resolved as a binary YES or NO answer. This dataset contains
            871 markets and a trading volume of $19 billion. I also obtained
            price history for each market, snapshotting prices at one week,
            three months and six months prior to market close (or as close as
            possible to those dates for markets with shorter lifespans). For
            each market, I calculated the Binary Brier Score at each time
            snapshot and at a composite all time snapshots. The Binary Brier
            Score is a measure of the accuracy of probabilistic predictions
            defined as:
        </p>
        <div class="flex justify-center my-4">
            <code
                class="text-lg bg-slate-100 dark:bg-slate-800 px-4 py-2 rounded text-sm font-mono"
                ><i> BS = (f - o)² </i></code
            >
        </div>
        <p>
            Where <code
                class="bg-slate-100 dark:bg-slate-800 px-1 rounded font-mono"
                >f</code
            >
            is the forecasted probability and
            <code class="bg-slate-100 dark:bg-slate-800 px-1 rounded font-mono"
                >o</code
            > is the actual outcome (1 for YES, 0 for NO). A lower Brier Score indicates
            a more accurate forecast and 50/50 predictions for all events would trend
            towards a Brier Score of 0.25 (assuming an even distribution between
            YES and NO resolutions). Binary Brier Scores for Polymarket consensus
            forecasts across the dataset is as follows:
        </p>
        <div
            class="container h-full mx-auto flex justify-center items-center leading-relaxed"
        >
            <div class="space-y-2 m-3 custom-container">
                <div class="table-container">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Time Horizon</th>
                                <th class="text-center">Brier Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>1 Week</td>
                                <td class="text-center">0.076297</td>
                            </tr>
                            <tr>
                                <td>1 Month</td>
                                <td class="text-center">0.095189</td>
                            </tr>
                            <tr>
                                <td>3 Months</td>
                                <td class="text-center">0.118513</td>
                            </tr>
                            <tr>
                                <td>6 Months</td>
                                <td class="text-center">0.124301</td>
                            </tr>
                            <tr class="font-bold border-t-2">
                                <td>Composite</td>
                                <td class="text-center">0.103575</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <p>
            As we would expect, prediction accuracy improves as the time horizon
            decreases. At time horizons of three months and less, Polymarket
            market consensus is in the superforecaster range (0.1 or less on the
            Binary Brier scale). This score is a good signal of predictive power
            within the market, with the caveat that questions may be easier or
            harder relative to other prediction market datasets. But is that
            predictive power evenly distributed? We can visualize the data to
            help evaluate how spiky performance is.
        </p>
        <h3 class="h3">Visualizing Polymarket Forecasts</h3>
        <p>
            In the science fiction TV show Westworld, an incredibly powerful AI
            predicts world events using data about human behaviour. The main
            interface for the AI, called "Rehoboam", is a circle of particles
            that shows spiky distortion as events begin to diverge from the
            predicted world state. While I can't recommend you watch Westworld
            past its second season in good conscience, the Rehoboam interface
            has stuck with me over time and I can't think of a more appropriate
            dataset to apply it to.
        </p>
        <div class="flex justify-center mt-0 flex-1 relative">
            <img
                class="max-h-[500px]"
                src={rehoboam}
                alt="rehoboam interface"
            />
        </div>
        <p>
            The below visualization displays Polymarket predictive performance
            in a Rehoboam-like interface. Each market is placed as a point on
            the circle, with points arranged by the semantic similarity of
            market questions. This approach groups like questions together. The
            particle spikes emanating out from the circle are based on areas of
            worse than 0.25 Brier Scores, showing instances where the world
            diverged from the market's predicted outcome. Projecting complex
            questions onto a 1D line is very lossy, so I also used similarity in
            Brier Score to contribute to ordering. Tooltips show market details
            including the count of days with active trades in the 90 days prior
            to each timestamp and there is a toggle to hide completely inactive
            markets that will not have a functional price consensus. You can
            access a full size version of this visualization <a
                href="/20260411_rehoboam.html"
                style="text-decoration: underline; color: lightblue;">here</a
            >.
        </p>
        <div class="w-full h-full overflow-hidden">
            <iframe
                src="/20260411_rehoboam.html"
                class="w-full h-full border-none ml-6 min-h-[600px]"
                title="Polymarket Rehoboam Visualization"
            ></iframe>
        </div>
        <p>
            While I love to look at this visualization, it is unfortunately a
            little unwieldy due to the diffusion of particles obscuring a
            complex distribution of points. An alternative visualization that
            preserves the details but still applies the same idea is below. This
            version shows each prediction as an arc out from the center of the
            circle based on the Brier Score. Incorrect predictions sit outside
            of the circle and show areas where the market forecast diverged from
            the outcome. You can access a full size version of this
            visualization <a
                href="/20260411_rehoboam_alternate.html"
                style="text-decoration: underline; color: lightblue;">here</a
            >.
        </p>
        <div class="w-full h-full overflow-hidden">
            <iframe
                src="/20260411_rehoboam_alternate.html"
                class="w-full h-full border-none ml-6 min-h-[600px]"
                title="Polymarket Rehoboam Visualization"
            ></iframe>
        </div>
        <p>
            I am most interested in predictions related to geopolitical and
            economic events, given that there are vanishingly few predictions
            related to my area of expertise, cyber domain events. Some
            observations that jump out at me include:
        </p>
        <ul class="list-disc ml-12">
            <li>
                There is bad performance around recent US military actions in
                Venezuela and Iran. Even at one week prior to market closes and
                with healthy liquidity, the consensus was against both military
                strikes and/or Maduro and Khamenei losing power. These failures
                may be self-fulfilling prophecies to some degree because the
                decapitation strikes used in both actions have some reliance on
                the element of surprise. There is a broader pattern of
                unpredictability surrounding Trump administration decisions
                (particular on tariffs) that speaks to the ascendancy of a more
                strongman conception of geopolitics that is harder to predict
                than the realist and economic models that have dominated
                previous decades.
            </li>
            <li>
                There are a number of failures to predict election-related
                outcomes, even at short time horizons. Election results
                predicted erroneously include Mamdani winning over Cuomo in the
                NYC mayoral Democratic primary, Nawrocki defeating Trzaskowski
                in the 2025 Polish presidential election, the unexpectedly
                favourable performance of D66 in the 2025 Netherlands general
                election and Dan defeating Simion in the 2025 Romanian
                presidential election. At a slightly longer time horizon of
                three months, Carney winning the Canadian Prime Ministership was
                also not predicted. There is not a clear pattern of left/right
                swings or anti-incumbency across these failures. I think these
                results are indicative of elections having a very small
                "Goldilocks-zone" of predictability compared to other types of
                events and being decided by difficult-to-predict campaign
                performance rather than candidate status quo at longer time
                horizons.
            </li>
            <li>
                The market failed to predict the downturn in cryptocurrency
                markets in late 2025 and overpredicted investment in new
                projects. My suspicion is that there is some positive selection
                bias for cryptocurrency enthusiasts among Polymarket
                participants that leads to myopia about the fundamentals of
                cryptocurrency markets. There may also be an element of actors
                attempting to create a self-fulfilling prophecy by placing bets
                that signal confidence in projects they already have a stake in.
                Finally, Polymarket may also be used to spread risk from
                external markets given it is not a closed system.
            </li>
        </ul>
        <p>
            This analysis suggests that Polymarket is an effective prediction
            market with minimal evidence of systemic failures in forecasting.
            The market mechanism shows a clear ability to normalize prices
            towards effective predictions that approach the accuracy of the best
            individual forecasters. If there are outcomes that are relevant to
            you being predicted on Polymarket, it <i>probably</i> is productive for
            you to follow those markets and have them inform your decision making
            if they are sufficiently liquid. This is not to say that you should participate
            in Polymarket yourself - there is a distinct chance that any market you
            participate in will become a vehicle to transfer money from you to insiders
            with privileged information. The same unfairness that makes Polymarket
            financially risky is probably a critical factor in its ability to forecast
            effectively.
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
