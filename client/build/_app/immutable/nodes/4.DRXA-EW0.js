import{s as Ne,e as a,a as o,c as s,b as Je,g as r,d as i,f as _e,h as p,B as Ye,i as Ze,j as n,l as Fe,n as Xe}from"../chunks/scheduler.DzZHELrM.js";import{S as et,i as tt,c as E,a as S,m as K,t as O,b as Q,d as U}from"../chunks/index.BWQOah18.js";import"../chunks/ProgressBar.svelte_svelte_type_style_lang.BrI0fzGN.js";import{C as R}from"../chunks/CodeBlock.Cigd8BMN.js";const nt=""+new URL("../assets/predictions.DQZxKzxl.png",import.meta.url).href;function ot(B){let c,e,h,Ce="@oj-sec",N,u,$e=`Evaluating Large Language Models as future event forecasters - Part
            Two: Performance & token sampling`,Y,_,Te="5 May 2024",Z,d,qe=`You can access a Juypter notebook (built for Colab) associated with
            this post <a href="https://github.com/oj-sec/blog/blob/main/notebooks/20240505.ipynb" style="text-decoration: underline; color: lightblue;">here</a>.`,F,m,Pe="Introduction",X,C,je=`We <a href="/blog/20240404" style="text-decoration: underline; color: lightblue;">left off</a> making predictions by repeatedly invoking a simple prompt and then
            examining the distribution of results. It&#39;s worth us examining what was
            going on in a little more detail. We got different results as the result
            of our temperature setting, which was configured to 0.7. Temperature
            in AI models is a measure of how stochastic our outputs are. At a temperature
            of zero, a model will always generate the most likely token, resulting
            in deterministic output. As we increase temperature, token selection
            becomes increasingly random, resulting in more diverse outputs. Temperature
            is commonly set in the range of 0.7-1.0 and higher temperature is often
            perceived by users as higher model intelligence and creativity.`,ee,$,Le=`We were using temperature to approximate the model's tendency to
            choose particular predictions over many samples. But we can achieve
            the same result more directly, by examining the probabilities of
            options straight from the model. This will give us a performance
            speedup equal to the number of samples we intend to take. While
            optimising a system so early is generally a mistake, its hard to
            leave a straightforward 100x or greater performance improvement on
            the table. To understand how we can peer directly into the model's
            probabilities, we'll need to understand tokenisation.`,te,g,Me="Tokenisation",ne,T,He=`Language models operate on text fragments called tokens, both when
            consuming prompts and generating output. Tokens are semantic
            fragments of written text, typically at the word and sub-word level,
            but sometimes down to the character level. We can tokenise some text
            as an example by directly accessing the tokeniser inside the model
            object maintained for us by Guidance:`,oe,f,ie,q,We="We get back the following output:",ae,b,se,P,Ie=`We can observe a mixture of whole and partial words in the output.
            We can also observe that words commonly have a leading space, which
            may be important for us to account for in some circumstances. A
            common rule of thumb for evaluating how many tokens are present in a
            particular string is that there are approximately four characters
            per token and approximately one token for every 0.75 words.`,re,v,ze="A note on how Guidance handles tokens",le,j,Ae=`When we provide Guidance with a regular expression, possible tokens
            are evaluated against the constraining regular expression and
            discarded if they do not match. This functioning is critical for us
            to understand, because we might otherwise assume that the generation
            is evaluated in larger chunks, like whole words or entire phrases.
            This misunderstanding can result in unexpected behaviour due to the
            model starting to generate a coherent answer that is consistent with
            the start of an incoherent option.`,pe,L,Ge='We can borrow an example from Guidance&#39;s Github <a href="https://github.com/guidance-ai/guidance/issues/564" style="text-decoration: underline; color: lightblue;">issues</a>:',ce,x,he,M,De=`Which gives us the perplexing output <span class="pre p-1">&quot;skill&quot;</span>
            rather than <span class="pre p-1">&quot;cloud&quot;</span>. The model was
            trying to generate a reasonable answer (<span class="pre p-1">&quot;skies&quot;</span>) that collided with the invalid
            <span class="pre p-1">&quot;skill&quot;</span> option. Once the model started
            generating, it could only output
            <span class="pre p-1">&quot;skill&quot;</span> despite the low coherence of the
            answer. As noted by a Guidance contributor, we can address this particular
            case by putting the options directly into the prompt to provide some
            context. We&#39;ll cover another pattern for addressing this issue in the
            next part in this series, but it is critical for us to understand that
            Guidance&#39;s constraints are at the naive token level so that we don&#39;t
            expect contextual evaluation where none is occurring.`,ue,k,Be="Sampling token probabilities directly",de,H,Ee=`With a background on tokens, we can have a look at directly
            accessing the probability of a particular output. Because generation
            happens token by token, we need to evaluate the probability of each
            token in sequence. These token-specific sequential probabilities
            commonly called the "logprobs" of tokens, defined as log(p) where p
            is the probability of the token occurring given the preceding
            tokens, both generated and prompt. We're going to stick with
            straight probabilities today, but if you shift this paradigm to a
            different stack, including OpenAI APIs, logprobs is the term to look
            for.`,me,W,Se=`Unfortunately, Guidance&#39;s interface for accessing logits is fairly
            nascent, so we need to implement the method for accessing
            probabilities ourselves. To keep things from getting too complex for
            now, we can cheat by pregenerating the <span class="pre p-1">&quot;0.&quot;</span> for predictions and instead just evaluating the probabilities of tokens
            in the tenths place. We can access the token logprobs with the following
            code:`,ge,w,fe,I,Ke="Which gives us the following results:",be,y,ve,z,Oe=`Here we have shortcut directly to the actual probability
            distribution for the tenths place in our prediction space. In
            effect, this is an instant 100x speedup relative to our previous
            approach of using temperature to iteratively explore this
            distribution by repeated generations. While we've kept things simple
            to demonstrate the concept, this approach can be extrapolated to
            multi-token generation by stepping the model through each token.`,xe,A,Qe=`This is a powerful capability with broad applicability to LLM tasks.
            If we're working on a jailbreak technique, we can evaluate exactly
            how likely it is to occur to properly assess its risk. If we're
            working on classification, we can evaluate the confidence of a given
            prediction. If we're performing finetuning or prompt engineering, we
            get granular insight into whether we're getting hotter or colder as
            we make changes.`,ke,G,Ue=`To finish up, we can repeat our proof of concept showing the actual
            probability distributions for previous proof of concept predictions.
            A little interestingly, the model is less emphatic on both
            predictions than our previous small sample size suggested.`,we,D,Re,V,ye,Ve;return f=new R({props:{language:"python",code:`# create a text string to tokenise
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
print(tokens)`}}),b=new R({props:{language:"plaintext",code:`[1, 14683, 6591, 6273, 298, 347, 2936, 298, 1221, 562, 605, 11143, 298, 2268, 28723]
['', ' Mind', 'sets', ' tend', ' to', ' be', ' quick', ' to', ' form', ' but', ' res', 'istant', ' to', ' change', '.']`}}),x=new R({props:{language:"python",code:`from guidance import models, select

# load the model
llm = models.LlamaCpp("./models/mistral-7b-openorca.Q4_K_M.gguf", n_gpu_layers=20, n_ctx=4096) 

# demonstrate bad generation - note that select() functions identically to a regex of the form "(cloud|skill)"
llm + 'A word very similar to "sky" is "' + select(["cloud","skill"])`}}),w=new R({props:{language:"python",code:`from guidance import models, gen
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
print(json.dumps(option_probs, indent=4))`}}),y=new R({props:{language:"plaintext",code:`The highest probability option in the tenths place is: 2
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
}`}}),{c(){c=a("div"),e=a("div"),h=a("h1"),h.textContent=Ce,N=o(),u=a("h2"),u.textContent=$e,Y=o(),_=a("p"),_.textContent=Te,Z=o(),d=a("p"),d.innerHTML=qe,F=o(),m=a("h3"),m.textContent=Pe,X=o(),C=a("p"),C.innerHTML=je,ee=o(),$=a("p"),$.textContent=Le,te=o(),g=a("h3"),g.textContent=Me,ne=o(),T=a("p"),T.textContent=He,oe=o(),E(f.$$.fragment),ie=o(),q=a("p"),q.textContent=We,ae=o(),E(b.$$.fragment),se=o(),P=a("p"),P.textContent=Ie,re=o(),v=a("h3"),v.textContent=ze,le=o(),j=a("p"),j.textContent=Ae,pe=o(),L=a("p"),L.innerHTML=Ge,ce=o(),E(x.$$.fragment),he=o(),M=a("p"),M.innerHTML=De,ue=o(),k=a("h3"),k.textContent=Be,de=o(),H=a("p"),H.textContent=Ee,me=o(),W=a("p"),W.innerHTML=Se,ge=o(),E(w.$$.fragment),fe=o(),I=a("p"),I.textContent=Ke,be=o(),E(y.$$.fragment),ve=o(),z=a("p"),z.textContent=Oe,xe=o(),A=a("p"),A.textContent=Qe,ke=o(),G=a("p"),G.textContent=Ue,we=o(),D=a("img"),this.h()},l(l){c=s(l,"DIV",{class:!0});var J=Je(c);e=s(J,"DIV",{class:!0});var t=Je(e);h=s(t,"H1",{class:!0,"data-svelte-h":!0}),r(h)!=="svelte-aomebi"&&(h.textContent=Ce),N=i(t),u=s(t,"H2",{class:!0,"data-svelte-h":!0}),r(u)!=="svelte-1ad4dy8"&&(u.textContent=$e),Y=i(t),_=s(t,"P",{"data-svelte-h":!0}),r(_)!=="svelte-bomvs2"&&(_.textContent=Te),Z=i(t),d=s(t,"P",{class:!0,"data-svelte-h":!0}),r(d)!=="svelte-1g3dbvv"&&(d.innerHTML=qe),F=i(t),m=s(t,"H3",{class:!0,"data-svelte-h":!0}),r(m)!=="svelte-fwvbau"&&(m.textContent=Pe),X=i(t),C=s(t,"P",{"data-svelte-h":!0}),r(C)!=="svelte-18k5reb"&&(C.innerHTML=je),ee=i(t),$=s(t,"P",{"data-svelte-h":!0}),r($)!=="svelte-xdxav7"&&($.textContent=Le),te=i(t),g=s(t,"H3",{class:!0,"data-svelte-h":!0}),r(g)!=="svelte-1oc4nk8"&&(g.textContent=Me),ne=i(t),T=s(t,"P",{"data-svelte-h":!0}),r(T)!=="svelte-95h6ll"&&(T.textContent=He),oe=i(t),S(f.$$.fragment,t),ie=i(t),q=s(t,"P",{"data-svelte-h":!0}),r(q)!=="svelte-1c6wsvm"&&(q.textContent=We),ae=i(t),S(b.$$.fragment,t),se=i(t),P=s(t,"P",{"data-svelte-h":!0}),r(P)!=="svelte-j99q6r"&&(P.textContent=Ie),re=i(t),v=s(t,"H3",{class:!0,"data-svelte-h":!0}),r(v)!=="svelte-1q161gx"&&(v.textContent=ze),le=i(t),j=s(t,"P",{"data-svelte-h":!0}),r(j)!=="svelte-x3jlge"&&(j.textContent=Ae),pe=i(t),L=s(t,"P",{"data-svelte-h":!0}),r(L)!=="svelte-1pevgl0"&&(L.innerHTML=Ge),ce=i(t),S(x.$$.fragment,t),he=i(t),M=s(t,"P",{"data-svelte-h":!0}),r(M)!=="svelte-td0agc"&&(M.innerHTML=De),ue=i(t),k=s(t,"H3",{class:!0,"data-svelte-h":!0}),r(k)!=="svelte-w3deqj"&&(k.textContent=Be),de=i(t),H=s(t,"P",{"data-svelte-h":!0}),r(H)!=="svelte-hnqmto"&&(H.textContent=Ee),me=i(t),W=s(t,"P",{"data-svelte-h":!0}),r(W)!=="svelte-kkoamv"&&(W.innerHTML=Se),ge=i(t),S(w.$$.fragment,t),fe=i(t),I=s(t,"P",{"data-svelte-h":!0}),r(I)!=="svelte-1tls2in"&&(I.textContent=Ke),be=i(t),S(y.$$.fragment,t),ve=i(t),z=s(t,"P",{"data-svelte-h":!0}),r(z)!=="svelte-1bc9j38"&&(z.textContent=Oe),xe=i(t),A=s(t,"P",{"data-svelte-h":!0}),r(A)!=="svelte-1yjvm5r"&&(A.textContent=Qe),ke=i(t),G=s(t,"P",{"data-svelte-h":!0}),r(G)!=="svelte-165w2i3"&&(G.textContent=Ue),we=i(t),D=s(t,"IMG",{src:!0,alt:!0}),t.forEach(_e),J.forEach(_e),this.h()},h(){p(h,"class","h1 text-center mb-12"),p(u,"class","h2"),p(d,"class","card variant-filled-ghost p-4"),p(m,"class","h3"),p(g,"class","h3"),p(v,"class","h3"),p(k,"class","h3"),Ye(D.src,Re=nt)||p(D,"src",Re),p(D,"alt","Predictions"),p(e,"class","space-y-5 m-10 custom-container svelte-1bdawuf"),p(c,"class","container h-full mx-auto flex justify-center items-center leading-relaxed")},m(l,J){Ze(l,c,J),n(c,e),n(e,h),n(e,N),n(e,u),n(e,Y),n(e,_),n(e,Z),n(e,d),n(e,F),n(e,m),n(e,X),n(e,C),n(e,ee),n(e,$),n(e,te),n(e,g),n(e,ne),n(e,T),n(e,oe),K(f,e,null),n(e,ie),n(e,q),n(e,ae),K(b,e,null),n(e,se),n(e,P),n(e,re),n(e,v),n(e,le),n(e,j),n(e,pe),n(e,L),n(e,ce),K(x,e,null),n(e,he),n(e,M),n(e,ue),n(e,k),n(e,de),n(e,H),n(e,me),n(e,W),n(e,ge),K(w,e,null),n(e,fe),n(e,I),n(e,be),K(y,e,null),n(e,ve),n(e,z),n(e,xe),n(e,A),n(e,ke),n(e,G),n(e,we),n(e,D),V=!0,ye||(Ve=Fe(h,"click",B[0]),ye=!0)},p:Xe,i(l){V||(O(f.$$.fragment,l),O(b.$$.fragment,l),O(x.$$.fragment,l),O(w.$$.fragment,l),O(y.$$.fragment,l),V=!0)},o(l){Q(f.$$.fragment,l),Q(b.$$.fragment,l),Q(x.$$.fragment,l),Q(w.$$.fragment,l),Q(y.$$.fragment,l),V=!1},d(l){l&&_e(c),U(f),U(b),U(x),U(w),U(y),ye=!1,Ve()}}}function it(B){window.location.href=B}function at(B){return[()=>it("/")]}class ct extends et{constructor(c){super(),tt(this,c,at,ot,Ne,{})}}export{ct as component};
