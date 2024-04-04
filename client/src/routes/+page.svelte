<script>

    function navigate(url) {
        window.location.href = url;
    }

</script>

<div class="container h-full mx-auto flex justify-center items-center">
	<div class="space-y-5 m-10">
		<h1 class="h1 text-center mb-0" on:click={() => navigate("/")}>@oj-sec</h1>
        <p class="text-sm text-center mb-20"><img >Oliver Smith<br>
        security | intelligence analysis | artificial intelligence</p>
        <div class="card variant-filled-surface p-4 my-6 card-hover" on:click={() => navigate("/blog/20240404")}>
            <header class="card-header"><h3 class="text-xl">Evaluating Large Language Models as future event forecasters - Part One</h3></header>
            <footer class="card-footer mt-2">4 April 2024</footer>
        </div>
	</div>
</div>