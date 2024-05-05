import{s as Re,e as a,a as o,c as i,b as Oe,g as r,d as s,f as we,h as p,z as Ve,i as Je,j as n,l as Ne,n as Ye}from"../chunks/scheduler.DhbVNR2E.js";import{S as Ze,i as Fe,c as E,a as S,m as B,t as K,b as Q,d as U}from"../chunks/index.BswMgjpc.js";import"../chunks/ProgressBar.svelte_svelte_type_style_lang.Cz8GQXdp.js";import{C as O}from"../chunks/CodeBlock.CEdCFRyr.js";const Xe=""+new URL("../assets/predictions.DQZxKzxl.png",import.meta.url).href;function et(D){let c,e,h,_e="@oj-sec",J,u,ye=`Evaluating Large Language Models as future event forecasters - Part
            Two: Performance & token sampling`,N,y,Ce="5 May 2024",Y,d,$e=`You can access a Juypter notebook (built for Colab) associated with
            this post <a href="https://github.com/oj-sec/blog/blob/main/notebooks/20240505.ipynb" style="text-decoration: underline; color: lightblue;">here</a>.`,Z,m,Te="Introduction",F,C,qe=`We <a href="/blog/20240404" style="text-decoration: underline; color: lightblue;">left off</a> making predictions by repeatedly invoking a simple prompt and then
            examining the distribution of results. It&#39;s worth us examining what was
            going on in a little more detail. We got different results as the result
            of our temperature setting, which was configured to 0.7. Temperature
            in AI models is a measure of how stochastic our outputs are. At a temperature
            of zero, a model will always generate the most likely token, resulting
            in deterministic output. As we increase temperature, token selection
            becomes increasingly random, resulting in more diverse outputs. Temperature
            is commonly set in the range of 0.7-1.0 and higher temperature is often
            perceived by users as higher model intelligence and creativity.`,X,$,Pe=`We were using temperature to approximate the model's tendency to
            choose particular predictions over many samples. But we can achieve
            the same result more directly, by examining the probabilities of
            options straight from the model. This will give us a performance
            speedup equal to the number of samples we intend to take. While
            optimising a system so early is generally a mistake, its hard to
            leave a straightforward 100x or greater performance improvement on
            the table. To understand how we can peer directly into the model's
            probabilities, we'll need to understand tokenisation.`,ee,g,je="Tokenisation",te,T,He=`Language models operate on text fragments called tokens, both when
            consuming prompts and generating output. Tokens are semantic
            fragments of written text, typically at the word and sub-word level,
            but sometimes down to the character level. We can tokenise some text
            as an example by directly accessing the tokeniser inside the model
            object maintained for us by Guidance:`,ne,f,oe,q,Me="We get back the following output:",se,b,ae,P,Le=`We can observe a mixture of whole and partial words in the output.
            We can also observe that words commonly have a leading space, which
            may be important for us to account for in some circumstances. A
            common rule of thumb for evaluating how many tokens are present in a
            particular string is that there are approximately four characters
            per token and approximately one token for every 0.75 words.`,ie,v,We="A note on how Guidance handles tokens",re,j,ze=`When we provide Guidance with a regular expression, possible tokens
            are evaluated against the constraining regular expression and
            discarded if they do not match. This functioning is critical for us
            to understand, because we might otherwise assume that the generation
            is evaluated in larger chunks, like whole words or entire phrases.
            This misunderstanding can result in unexpected behaviour due to the
            model starting to generate a coherent answer that is consistent with
            the start of an incoherent option.`,le,H,Ge='We can borrow an example from Guidance&#39;s Github <a href="https://github.com/guidance-ai/guidance/issues/564" style="text-decoration: underline; color: lightblue;">issues</a>:',pe,x,ce,M,Ie=`Which gives us the perplexing output <span class="pre p-1">&quot;skill&quot;</span>
            rather than <span class="pre p-1">&quot;cloud&quot;</span>. The model was
            trying to generate a reasonable answer (<span class="pre p-1">&quot;skies&quot;</span>) that collided with the invalid
            <span class="pre p-1">&quot;skill&quot;</span> option. Once the model started
            generating, it could only output
            <span class="pre p-1">&quot;skill&quot;</span> despite the low coherence of the
            answer. As noted by one of the Guidance devs, we can address this particular
            case by putting the options directly into the prompt to provide some
            context. We&#39;ll cover another pattern for addressing this issue in the
            next part in this series, but it is critical for us to understand that
            Guidance&#39;s constraints are at the naive token level so that we don&#39;t
            expect contextual evaluation where none is occurring.`,he,k,Ae="Sampling token probabilities directly",ue,L,De=`With a background on tokens, we can have a look at directly
            accessing the probability of a particular output. Because generation
            happens token by token, when evaluating a sequence, we're evaluating
            the probability of each token in the entire sequence. These
            sequential probabilities are sometimes called the "logprobs" of
            tokens, defined as log(p) where p is the probability of the token
            occurring given the preceding tokens, both generated and prompt.`,de,W,Ee=`Unfortunately, Guidance&#39;s interface for accessing logits is fairly
            nascent, so we need to implement the method for accessing
            probabilities ourselves. To keep things from getting too complex for
            now, we can cheat by pregenerating the <span class="pre p-1">&quot;0.&quot;</span> for predictions and instead just evaluating the probabilities of tokens
            in the tenths place. We can access the token logprobs with the following
            code:`,me,w,ge,z,Se="Which gives us the following results:",fe,_,be,G,Be=`Here we have shortcut directly to the actual probability
            distribution for the tenths place in our prediction space. In
            effect, this is an instant 100x speedup relative to our previous
            approach of using temperature to iteratively explore this
            distribution by repeated generations. While we've kept things simple
            to demonstrate the concept, this approach can be extrapolated to
            multi-token generation by stepping the model through each token.`,ve,I,Ke=`To finish up, we can repeat our proof of concept showing the actual
            probability distributions for previous proof of concept predictions.
            A little interestingly, the model is less emphatic on both
            predictions than our previous small sample size sugguested.`,xe,A,Qe,R,ke,Ue;return f=new O({props:{language:"python",code:`# create a text string to tokenise
string = "Mindsets tend to be quick to form but resistant to change."

# generate the tokens by accessing the model tokeniser
tokens_encoded = llm.engine.model_obj.tokenize(str.encode(string))

# decode the tokens
tokens = []
for token in tokens_encoded:
    if token:
        tokens.append(llm.engine.model_obj.detokenize([token]).decode("utf-8", errors="replace"))

# show results
print(tokens_encoded)
print(tokens)`}}),b=new O({props:{language:"plaintext",code:`[1, 14683, 6591, 6273, 298, 347, 2936, 298, 1221, 562, 605, 11143, 298, 2268, 28723]
['', ' Mind', 'sets', ' tend', ' to', ' be', ' quick', ' to', ' form', ' but', ' res', 'istant', ' to', ' change', '.']`}}),x=new O({props:{language:"python",code:`from guidance import models, select

# load the model
llm = models.LlamaCpp("./models/mistral-7b-openorca.Q4_K_M.gguf", n_gpu_layers=20, n_ctx=4096) 

# demonstrate bad generation - note that select() functions identically to a regex of the form "(cloud|skill)"
llm + 'A word very similar to "sky" is "' + select(["cloud","skill"])`}}),w=new O({props:{language:"python",code:`from guidance import models, gen
import llama_cpp
import torch
import math
import json

# load the model
llm = models.LlamaCpp("./models/mistral-7b-openorca.Q4_K_M.gguf", compute_log_probs=True, n_gpu_layers=20, n_ctx=4096) 

# define a regular expression to match a single number
output_regex = r"\\d"

# define our prompt - note that we've added a "0." force output to just examine the tenths place
prompt = 'Predict the likelihood of the following outcome on a scale from 0.00 to 1.00, with 0.00 meaning the event is impossible and 1.00 meaning the event is certain to occur: "Donald Trump will win the 2024 US election."\\nPREDICTION:0.'

# run constrained inference - note that we have set temperature to zero
output = llm + prompt + gen(name="response", regex=output_regex, max_tokens=1, temperature=0.0)

# define the options we want to check the probs for
options = [f"{n}" for n in range(0,10)] 

# retrieve the logits from the model object
logits = llama_cpp.llama_get_logits(llm.engine.model_obj.ctx)

# tokenize our options
option_tokens = [llm.engine.model_obj.tokenize(str.encode(o)) for o in options]

# retrieve just the option token, discarding the <s> added by the tokenizer
option_tokens = [o[2] for o in option_tokens] 

# retrieve the logits for the option
option_logits = [logits[o] for o in option_tokens]

# convert the logits into probabilities
option_probs = torch.softmax(torch.tensor(option_logits), dim=0)

# typecast to floats 
option_probs = [float(o) for o in option_probs]

# zip the options and probabilities together
option_probs = dict(zip(options, option_probs))

# get the top token
top_token = max(option_probs, key=option_probs.get)

# print results
print(f"The highest probability option in the tenths place is: {top_token.strip()}")
print("The probability distribution for the tenths place is: ")
print(json.dumps(option_probs, indent=4))`}}),_=new O({props:{language:"plaintext",code:`The highest probability option in the tenths place is: 2
The probability distribution for the tenths place is: 
{
    "0": 0.15142710506916046,
    "1": 0.15139013528823853,
    "2": 0.18116815388202667,
    "3": 0.16766728460788727,
    "4": 0.12874439358711243,
    "5": 0.12978236377239227,
    "6": 0.0467846542596817,
    "7": 0.02292967587709427,
    "8": 0.010299457237124443,
    "9": 0.009806782938539982
}`}}),{c(){c=a("div"),e=a("div"),h=a("h1"),h.textContent=_e,J=o(),u=a("h2"),u.textContent=ye,N=o(),y=a("p"),y.textContent=Ce,Y=o(),d=a("p"),d.innerHTML=$e,Z=o(),m=a("h3"),m.textContent=Te,F=o(),C=a("p"),C.innerHTML=qe,X=o(),$=a("p"),$.textContent=Pe,ee=o(),g=a("h3"),g.textContent=je,te=o(),T=a("p"),T.textContent=He,ne=o(),E(f.$$.fragment),oe=o(),q=a("p"),q.textContent=Me,se=o(),E(b.$$.fragment),ae=o(),P=a("p"),P.textContent=Le,ie=o(),v=a("h3"),v.textContent=We,re=o(),j=a("p"),j.textContent=ze,le=o(),H=a("p"),H.innerHTML=Ge,pe=o(),E(x.$$.fragment),ce=o(),M=a("p"),M.innerHTML=Ie,he=o(),k=a("h3"),k.textContent=Ae,ue=o(),L=a("p"),L.textContent=De,de=o(),W=a("p"),W.innerHTML=Ee,me=o(),E(w.$$.fragment),ge=o(),z=a("p"),z.textContent=Se,fe=o(),E(_.$$.fragment),be=o(),G=a("p"),G.textContent=Be,ve=o(),I=a("p"),I.textContent=Ke,xe=o(),A=a("img"),this.h()},l(l){c=i(l,"DIV",{class:!0});var V=Oe(c);e=i(V,"DIV",{class:!0});var t=Oe(e);h=i(t,"H1",{class:!0,"data-svelte-h":!0}),r(h)!=="svelte-aomebi"&&(h.textContent=_e),J=s(t),u=i(t,"H2",{class:!0,"data-svelte-h":!0}),r(u)!=="svelte-1ad4dy8"&&(u.textContent=ye),N=s(t),y=i(t,"P",{"data-svelte-h":!0}),r(y)!=="svelte-bomvs2"&&(y.textContent=Ce),Y=s(t),d=i(t,"P",{class:!0,"data-svelte-h":!0}),r(d)!=="svelte-1g3dbvv"&&(d.innerHTML=$e),Z=s(t),m=i(t,"H3",{class:!0,"data-svelte-h":!0}),r(m)!=="svelte-fwvbau"&&(m.textContent=Te),F=s(t),C=i(t,"P",{"data-svelte-h":!0}),r(C)!=="svelte-18k5reb"&&(C.innerHTML=qe),X=s(t),$=i(t,"P",{"data-svelte-h":!0}),r($)!=="svelte-xdxav7"&&($.textContent=Pe),ee=s(t),g=i(t,"H3",{class:!0,"data-svelte-h":!0}),r(g)!=="svelte-1oc4nk8"&&(g.textContent=je),te=s(t),T=i(t,"P",{"data-svelte-h":!0}),r(T)!=="svelte-95h6ll"&&(T.textContent=He),ne=s(t),S(f.$$.fragment,t),oe=s(t),q=i(t,"P",{"data-svelte-h":!0}),r(q)!=="svelte-1c6wsvm"&&(q.textContent=Me),se=s(t),S(b.$$.fragment,t),ae=s(t),P=i(t,"P",{"data-svelte-h":!0}),r(P)!=="svelte-j99q6r"&&(P.textContent=Le),ie=s(t),v=i(t,"H3",{class:!0,"data-svelte-h":!0}),r(v)!=="svelte-1q161gx"&&(v.textContent=We),re=s(t),j=i(t,"P",{"data-svelte-h":!0}),r(j)!=="svelte-x3jlge"&&(j.textContent=ze),le=s(t),H=i(t,"P",{"data-svelte-h":!0}),r(H)!=="svelte-1pevgl0"&&(H.innerHTML=Ge),pe=s(t),S(x.$$.fragment,t),ce=s(t),M=i(t,"P",{"data-svelte-h":!0}),r(M)!=="svelte-1mr4o22"&&(M.innerHTML=Ie),he=s(t),k=i(t,"H3",{class:!0,"data-svelte-h":!0}),r(k)!=="svelte-w3deqj"&&(k.textContent=Ae),ue=s(t),L=i(t,"P",{"data-svelte-h":!0}),r(L)!=="svelte-qz41je"&&(L.textContent=De),de=s(t),W=i(t,"P",{"data-svelte-h":!0}),r(W)!=="svelte-kkoamv"&&(W.innerHTML=Ee),me=s(t),S(w.$$.fragment,t),ge=s(t),z=i(t,"P",{"data-svelte-h":!0}),r(z)!=="svelte-1tls2in"&&(z.textContent=Se),fe=s(t),S(_.$$.fragment,t),be=s(t),G=i(t,"P",{"data-svelte-h":!0}),r(G)!=="svelte-1bc9j38"&&(G.textContent=Be),ve=s(t),I=i(t,"P",{"data-svelte-h":!0}),r(I)!=="svelte-163b5tk"&&(I.textContent=Ke),xe=s(t),A=i(t,"IMG",{src:!0,alt:!0}),t.forEach(we),V.forEach(we),this.h()},h(){p(h,"class","h1 text-center mb-12"),p(u,"class","h2"),p(d,"class","card variant-filled-ghost p-4"),p(m,"class","h3"),p(g,"class","h3"),p(v,"class","h3"),p(k,"class","h3"),Ve(A.src,Qe=Xe)||p(A,"src",Qe),p(A,"alt","Predictions"),p(e,"class","space-y-5 m-10 custom-container svelte-1bdawuf"),p(c,"class","container h-full mx-auto flex justify-center items-center leading-relaxed")},m(l,V){Je(l,c,V),n(c,e),n(e,h),n(e,J),n(e,u),n(e,N),n(e,y),n(e,Y),n(e,d),n(e,Z),n(e,m),n(e,F),n(e,C),n(e,X),n(e,$),n(e,ee),n(e,g),n(e,te),n(e,T),n(e,ne),B(f,e,null),n(e,oe),n(e,q),n(e,se),B(b,e,null),n(e,ae),n(e,P),n(e,ie),n(e,v),n(e,re),n(e,j),n(e,le),n(e,H),n(e,pe),B(x,e,null),n(e,ce),n(e,M),n(e,he),n(e,k),n(e,ue),n(e,L),n(e,de),n(e,W),n(e,me),B(w,e,null),n(e,ge),n(e,z),n(e,fe),B(_,e,null),n(e,be),n(e,G),n(e,ve),n(e,I),n(e,xe),n(e,A),R=!0,ke||(Ue=Ne(h,"click",D[0]),ke=!0)},p:Ye,i(l){R||(K(f.$$.fragment,l),K(b.$$.fragment,l),K(x.$$.fragment,l),K(w.$$.fragment,l),K(_.$$.fragment,l),R=!0)},o(l){Q(f.$$.fragment,l),Q(b.$$.fragment,l),Q(x.$$.fragment,l),Q(w.$$.fragment,l),Q(_.$$.fragment,l),R=!1},d(l){l&&we(c),U(f),U(b),U(x),U(w),U(_),ke=!1,Ue()}}}function tt(D){window.location.href=D}function nt(D){return[()=>tt("/")]}class rt extends Ze{constructor(c){super(),Fe(this,c,nt,et,Re,{})}}export{rt as component};
