<script>
	import '../app.postcss';

	// Highlight JS
	import hljs from 'highlight.js/lib/core';
	import 'highlight.js/styles/github-dark.css';
	import { storeHighlightJs } from '@skeletonlabs/skeleton';
	import json from 'highlight.js/lib/languages/json'; 
    import python from 'highlight.js/lib/languages/python';
    import plaintext from 'highlight.js/lib/languages/plaintext';
	hljs.registerLanguage('json', json); 
    hljs.registerLanguage('python', python);
    hljs.registerLanguage('plaintext', plaintext);
	storeHighlightJs.set(hljs);

	// Floating UI for Popups
	import { computePosition, autoUpdate, flip, shift, offset, arrow } from '@floating-ui/dom';
	import { storePopup } from '@skeletonlabs/skeleton';
	storePopup.set({ computePosition, autoUpdate, flip, shift, offset, arrow });

</script>

<slot />
