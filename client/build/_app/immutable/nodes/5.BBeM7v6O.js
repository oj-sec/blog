import{s as $t,e as i,a,c as s,b as Tt,g as r,d as o,f as Be,h as l,i as At,j as n,l as X,n as Pt,r as jt}from"../chunks/scheduler.DzZHELrM.js";import{S as Dt,i as Wt,c as te,a as ne,m as ae,t as oe,b as ie,d as se}from"../chunks/index.BWQOah18.js";import{g as Vt}from"../chunks/stores.1kU3aqDV.js";import"../chunks/ProgressBar.svelte_svelte_type_style_lang.BrI0fzGN.js";import{C as re}from"../chunks/CodeBlock.Cigd8BMN.js";const zt=""+new URL("../assets/diagram.C1XDZ9FI.png",import.meta.url).href,Mt=""+new URL("../assets/map1.ONeK109t.png",import.meta.url).href,It=""+new URL("../assets/map2.D1GzLClL.png",import.meta.url).href,Ht=""+new URL("../assets/map3.CenwWXUY.png",import.meta.url).href,qt=""+new URL("../assets/map4.tuAAvmcn.png",import.meta.url).href;function Rt(u){let d,e,h,le="@oj-sec",K,p,ce="Research aside - Hallucination detection & LLM explainability",P,f,Ge="5 July 2024",de,_,Ne=`You can access a Juypter notebook (built for Colab) associated with
            this post <a href="https://github.com/oj-sec/blog/blob/main/notebooks/20240705.ipynb" style="text-decoration: underline; color: lightblue;">here</a>.`,ue,y,Ye="Introduction",he,j,Qe=`The term "hallucination" is commonly used to describe factually
            inaccurate or incoherent outputs from generative AI models. Large
            Language Models (LLMs) are particularly prone to problematic
            hallucinations because instruct and chat fine tuning bias models
            towards producing confident-looking outputs and models lack any
            ability to introspect about confidence. Instruct and chat fine
            tuning are processes that convert a base LLM, which functions as an
            autocomplete engine that expands on a user's input, to the AI
            assistant, prompt-response/Q&A style outputs we're familiar with.`,pe,D,Xe=`Hallucinations can be a major barrier to deploying AI systems that
            can function without close human oversight and are an important
            consideration for almost every real-world LLM application. In this
            blog, I'm going to dive into hallucination detection and LLM
            explainability. This blog will involve examination of internal model
            states not made available through most inference backends, so we'll
            be using Transformers rather than our usual Guidance+LLama.cpp
            stack. The notebook and code samples provided will not interoperate
            with previously provided code.`,fe,x,Ke="Detecting hallucinated outputs",ge,W,Ze=`The idea that any given AI generated sequence can be classified on a
            binary between hallucinatory or factual is an oversimplification.
            The meaning encoded by language models can be thought of as a
            measure of the strength of the associativity between items in a
            sequence. There is no underlying concept of truth, it's all a
            spectrum of likelihoods - given this input, what is the likelihood
            of this output? In the real world, there are degrees of
            incorrectness that have important nuance - for example, an incorrect
            extrapolation from accurately-regurgitated facts probably poses a
            different risk than an outright fabrication.`,me,k,et=`<img src="${zt}" alt="Diagram"/>`,ve,V,tt=`LLMs' foundation in likelihoods gives us a clear path towards
            identifying hallucinations. We can assume that a coherent model will
            be less likely to generate a hallucination than a fact.
            Anthropomorphising, we can think of using the likelihood as a proxy
            to evaluate how "sure" of an answer the model is. This might not
            always map to factualness, but in my opinion, knowing that an LLM
            has a low confidence in its output is actually more useful. While a
            low confidence output might be factually correct by chance, we can
            infer that we've pushed the LLM to the limits of its knowledge and
            that we need to exercise caution around that topic. This method
            won't help us find very confident falsehoods, such as when a LLM was
            trained on a common misconception, but that class of hallucination
            is likely to be almost impossible to control for at inference time
            once it has been baked into a model during training.`,be,z,nt=`In June 2024, Oxford University researchers published an article in <i>Nature</i>
            titled
            <a href="https://www.nature.com/articles/s41586-024-07421-0" style="text-decoration: underline; color: lightblue;">&quot;Detecting hallucinations in large language models using
                semantic entropy&quot;</a>
            centered around this idea. The researchers proposed sampling outputs
            repeatedly and evaluating the spread (entropy) of outputs, with an added
            step of bucketing semantically similar answers. Combined with a system
            of extracting and checking standalone facts in larger generations, the
            proposed system had state-of-the-art confabulation detection rates in
            the 0.75-0.80 range depending on model.`,_e,R,at=`Solutions that involve repeated inference sampling are extremely
            costly, potentially prohibitively costly. We demonstrated in <a href="/blog/20240505" style="text-decoration: underline; color: lightblue;">the last blog</a>
            that we can substitute sampling with accessing logits and evaluating
            how likely each token is to be generated - the same idea theoretically
            holds here. Accessing logits under Transformers looks like this (note
            that we&#39;re using a Python object to encapsulate a bit more functionality
            than usual - check the
            <a href="https://github.com/oj-sec/blog/blob/main/notebooks/20240705.ipynb" style="text-decoration: underline; color: lightblue;">notebook</a> to see the full setup):`,ye,w,xe,S,ot="Let's instantiate the object and run a simple prompt:",ke,L,we,U,it="Which gives us the following output:",Le,C,Ce,F,st=`We inherit some additional complexity when we extend this idea over
            a whole string rather than just looking at a single token. Our
            lowest confidence token was the starting token, <span class="pre p-1">&quot;The&quot;</span>, which is is unlikely to be material to the meaning of the answer.
            It&#39;s also axiomatic that we&#39;re going to get smaller and smaller
            probabilities the longer our string gets.`,Te,O,rt=`Ultimately, what we want is a single metric than can give us a
            consistent read on confidence, regardless of the length and
            complexity of the input. I included a return type that takes the
            average of the individual tokens rather than their multiple to help
            control for variance in output length. I gave the metrics a test
            over a small sample of the TriviaQA dataset, one of the datasets
            used by the Oxford researchers.`,Me,E,lt=`Both metrics had comparable performance to identify confabulations,
            in the state of the art range of 0.75-0.80 (noting a small sample
            size). I used 0.5 as the confidence threshold for the overall
            probability, and 0.75 as the confidence threshold for average token
            probability. If an answer fell below the confidence threshold, I
            regarded it as likely wrong, if above the threshold likely correct.
            Taking the average of both metrics eked out a little extra
            performance.`,Ie,T,ct='<div class="space-y-2 m-3 custom-container svelte-1bdawuf"><div class="table-container"><table class="table table-hover"><thead><tr><th>Method</th> <th>Threshold</th> <th>Result</th></tr></thead> <tbody><tr><td>Total probability</td> <td class="text-center">0.5</td> <td class="text-center">0.78</td></tr> <tr><td>Average token probability</td> <td class="text-center">0.75</td> <td class="text-center">0.76</td></tr> <tr><td>Average of both metrics</td> <td class="text-center">0.625</td> <td class="text-center">0.81</td></tr></tbody></table></div></div>',He,J,dt=`These thresholds won't work for generations that aren't
            one-sentence/one-fact, but given that multi-fact generations almost
            certainly need to be decomposed to be evaluated, I think this is a
            performance friendly alternative to the semantic entropy sample and
            bucket technique and worth putting in the toolbox.`,qe,M,ut="Diving deeper",$e,B,ht=`Like all complex AI architectures, transformer-based models work by
            passing inputs through a series of hidden layers containing a
            network of neurons that activate to various degrees based features
            learned through training. The most advanced public research on LLM
            explainability is Anthropic&#39;s incredible
            <a href="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html" style="text-decoration: underline; color: lightblue;">&quot;Mapping the Mind of a Large Language Model&quot;</a>
            research, which demonstrates that LLMs coalesce human-comprehensible
            concepts in the activation states of hidden layers. These identifiable
            features can be modulated manually to change a model&#39;s behavior.`,Ae,G,pt=`This evokes some interesting questions around the topic of
            hallucinations. Is there a feature hidden in LLMs that is highly
            correlated with hallucinating? In humans, lying and creativity are
            both distinctive brain states. Is there a feature that is
            responsible for influencing the output "I don't know" that we could
            tune up to decrease confabulations?`,Pe,N,ft=`Unfortunately, Anthropic&#39;s research involved training custom sparse
            autoencoders that are beyond me to develop and train. We can get a
            glimpse into the internal state of models by extracting the hidden
            layers and visualising activations. We can access the hidden state
            through transformers by instantiating our model with
            <span class="pre p-1">output_hidden_states=True</span> and calling
            <span class="pre p-1">generate()</span> with
            <span class="pre p-1">return_dict_in_generate=True</span>. This
            gives us access to a <span class="pre p-1">hidden_states</span> array
            attribute on our generate output, where the first element is a tensor
            containing the hidden states for the entire context and each subsequent
            array element is a tensor containing the hidden state activations for
            each generated token. I visualised the average activations across the
            entire layer for each token with the below code:`,je,I,De,Y,gt="Which produces outputs that look like this:",We,H,mt='<h4 class="h4">&quot;What is the capital of Australia?&quot; - Factual generation</h4>',Ve,g,vt=`<img src="${Mt}" alt="Map 1"/>`,ze,q,bt=`<h4 class="h4">&quot;What is the target of Sotorasib?&quot; - Incomplete factual
                generation</h4>`,Re,m,_t=`<img src="${It}" alt="Map 2"/>`,Se,$,yt=`<h4 class="h4">&quot;Which was the first European country to abolish capital
                punishment?&quot; - Confabulation</h4>`,Ue,v,xt=`<img src="${Ht}" alt="Map 3"/>`,Fe,A,kt='<h4 class="h4">&quot;How did Jock die in Dallas?&quot; - Safety-related refusal</h4>',Oe,b,wt=`<img src="${qt}" alt="Map 4"/>`,Ee,Q,Lt=`We can make a few observations. Firstly, we have a very consistent
            average model states until the later layers, where things get more
            diverse. Secondly, we have very distinctive stripes of low average
            activation on punctuation and definite articles. A possible
            explanation for this is that these are indistinct "filler" tokens
            that do not require much activity within the model, but we'd really
            need to look at the individual states to understand further. At this
            resolution we can't see visual patterns that differentiate confident
            answers, confabulations and our refusal, but this isn't particularly
            surprising. We'd need to visualise neuron in each layer individually
            to see such subtle features. Each layer in the tiny
            Phi-3-mini-4k-instruct model I'm using has 3072 neurons for each of
            its 33 layers, so our odds of just eyeballing the features we're
            looking for are next to zero - we'd need to approach it as a
            statistical or deep learning task. Given the length of this blog
            already, that will have to be a topic for the future.`,Z,Je,Ct;return w=new re({props:{language:"python",code:`
    def generate_with_probability(
        self,
        prompt,
        response_prefix=None,
        max_tokens=200,
        temperature=0.0,
        round_to=6,
    ):
        """
        Method to generate a response with information about
        token probabilities.

        Args:
            prompt: str: The prompt to generate a response for.
            response_prefix: str: String to prefix the LLM's response.
            max_tokens: int: The maximum number of tokens to generate.
            temperature: float: The temperature to use for sampling.
            round_to: int: The number of decimal places to round to.

        Returns:
            str: The generated response.
            float: The total probability of the generated response.
            float: The average probability of each token in the generated response.
            list[tuple]: The individual token probabilities.
        """
        formatted_prompt = f"{
            self.system_prompt_start}{
            self.system_prompt}{
            self.prompt_suffix}{
                self.user_prompt_start}{prompt}{
                    self.prompt_suffix}{
                        self.assistant_prompt_start}"
        if response_prefix:
            formatted_prompt = formatted_prompt + response_prefix
        logging.info("LLMAgent generating response with probability for: %s.", prompt)
        inputs = self.processor(formatted_prompt, return_tensors="pt").to(self.device)
        generate_output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=bool(temperature > 0.0),
            temperature=temperature,
        )
        generate_ids = generate_output.sequences[:, inputs["input_ids"].shape[1] :]
        generated_sequence = generate_ids[0].cpu().numpy()
        generated_sequence = generated_sequence[:-2]  # Remove eos tokens
        response = self.processor.batch_decode(
            [generated_sequence],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        logging.info("  LLMAgent generated response: '%s'", response)
        logits = torch.stack(generate_output.scores, dim=1).cpu()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
        log_likelihood_for_gen = sum(
            log_probs[0, i, token_id].item()
            for i, token_id in enumerate(generated_sequence)
        )
        total_probability_for_gen = round(np.exp(log_likelihood_for_gen), round_to)
        individual_token_probs = []
        for i, token_id in enumerate(generated_sequence):
            token_prob = np.exp(log_probs[0, i, token_id].item())
            individual_token_probs.append(
                (self.processor.decode(token_id), round(token_prob, round_to))
            )
        average_token_propabiility = round(
            sum(token_prob for _, token_prob in individual_token_probs)
            / len(individual_token_probs),
            round_to,
        )
        return (
            response,
            total_probability_for_gen,
            average_token_propabiility,
            individual_token_probs,
        )`}}),L=new re({props:{language:"python",code:`lm = LLMAgent(
    system_prompt="You are a helpful AI assistant that provides correct information as concisely as possible.\\n"
)
response, total_probability, average_token_probability, individual_token_probs = lm.generate_with_probability(
    "What is the capital of Australia?"
)

print(f"Response generated:\\n{response}")
print(f"Total response probability:\\n{total_probability}")
print(f"Average token probability:\\n{average_token_probability}")
print("Individual token probabilities:")
for token in individual_token_probs:
    buff = 20 * " "
    buff = token[0] + buff[len(token[0]) :]
    print(f"{buff} : {token[1]}")`}}),C=new re({props:{language:"plaintext",code:`Response generated:
The capital of Australia is Canberra.
Total response probability:
0.841735
Average token probability:
0.982328
Individual token probabilities:
The                  : 0.845985
capital              : 0.999886
of                   : 0.997526
Australia            : 0.999978
is                   : 0.999964
Can                  : 0.999985
ber                  : 1.0
ra                   : 0.999999
.                    : 0.997632`}}),I=new re({props:{language:"python",code:`    def visualise_average_activations(self, outputs):
    """
    Method to visualise average activations per layer as a heatmap.
    """
    logging.info("LLMAgent visualising average activations for sequence.")
    tokens = [
        self.processor.decode(input_token) for input_token in outputs.sequences[0]
    ]
    average_activations = []
    for layer_states in outputs.hidden_states[0]:
        avg_activation = layer_states.squeeze(0).mean(dim=-1)
        average_activations.append(avg_activation)

    for layer_states in outputs.hidden_states[1:]:
        for i, layer_state in enumerate(layer_states):
            avg_activation = layer_state.squeeze(0).mean(dim=-1)
            average_activations[i] = torch.cat(
                [average_activations[i], avg_activation]
            )

    average_activations = torch.stack(average_activations, dim=1)
    figsize_x = max(12, len(outputs.hidden_states[0]) * 0.8)
    figsize_y = max(8, len(tokens) * 0.3)

    plt.figure(figsize=(figsize_x, figsize_y))
    sns.heatmap(
        average_activations.detach().cpu().numpy(),
        cmap="mako_r",
        xticklabels=[f"Layer {i}" for i in range(len(outputs.hidden_states[0]))],
        yticklabels=tokens,
        linecolor="lightgrey",
        linewidths=0.2,
        cbar=True,
    )
    plt.title("Average activation per layer per token")
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_bytes = buffer.read()
    plt.close()
    logging.info("LLMAgent visualised average activations for sequence.")
    return image_bytes`}}),{c(){d=i("div"),e=i("div"),h=i("h1"),h.textContent=le,K=a(),p=i("h2"),p.textContent=ce,P=a(),f=i("p"),f.textContent=Ge,de=a(),_=i("p"),_.innerHTML=Ne,ue=a(),y=i("h3"),y.textContent=Ye,he=a(),j=i("p"),j.textContent=Qe,pe=a(),D=i("p"),D.textContent=Xe,fe=a(),x=i("h3"),x.textContent=Ke,ge=a(),W=i("p"),W.textContent=Ze,me=a(),k=i("div"),k.innerHTML=et,ve=a(),V=i("p"),V.textContent=tt,be=a(),z=i("p"),z.innerHTML=nt,_e=a(),R=i("p"),R.innerHTML=at,ye=a(),te(w.$$.fragment),xe=a(),S=i("p"),S.textContent=ot,ke=a(),te(L.$$.fragment),we=a(),U=i("p"),U.textContent=it,Le=a(),te(C.$$.fragment),Ce=a(),F=i("p"),F.innerHTML=st,Te=a(),O=i("p"),O.textContent=rt,Me=a(),E=i("p"),E.textContent=lt,Ie=a(),T=i("div"),T.innerHTML=ct,He=a(),J=i("p"),J.textContent=dt,qe=a(),M=i("h3"),M.textContent=ut,$e=a(),B=i("p"),B.innerHTML=ht,Ae=a(),G=i("p"),G.textContent=pt,Pe=a(),N=i("p"),N.innerHTML=ft,je=a(),te(I.$$.fragment),De=a(),Y=i("p"),Y.textContent=gt,We=a(),H=i("div"),H.innerHTML=mt,Ve=a(),g=i("div"),g.innerHTML=vt,ze=a(),q=i("div"),q.innerHTML=bt,Re=a(),m=i("div"),m.innerHTML=_t,Se=a(),$=i("div"),$.innerHTML=yt,Ue=a(),v=i("div"),v.innerHTML=xt,Fe=a(),A=i("div"),A.innerHTML=kt,Oe=a(),b=i("div"),b.innerHTML=wt,Ee=a(),Q=i("p"),Q.textContent=Lt,this.h()},l(c){d=s(c,"DIV",{class:!0});var ee=Tt(d);e=s(ee,"DIV",{class:!0});var t=Tt(e);h=s(t,"H1",{class:!0,"data-svelte-h":!0}),r(h)!=="svelte-aomebi"&&(h.textContent=le),K=o(t),p=s(t,"H2",{class:!0,"data-svelte-h":!0}),r(p)!=="svelte-182btrv"&&(p.textContent=ce),P=o(t),f=s(t,"P",{"data-svelte-h":!0}),r(f)!=="svelte-i2bt81"&&(f.textContent=Ge),de=o(t),_=s(t,"P",{class:!0,"data-svelte-h":!0}),r(_)!=="svelte-8yq1d5"&&(_.innerHTML=Ne),ue=o(t),y=s(t,"H3",{class:!0,"data-svelte-h":!0}),r(y)!=="svelte-fwvbau"&&(y.textContent=Ye),he=o(t),j=s(t,"P",{"data-svelte-h":!0}),r(j)!=="svelte-1dwrnj4"&&(j.textContent=Qe),pe=o(t),D=s(t,"P",{"data-svelte-h":!0}),r(D)!=="svelte-8lcrty"&&(D.textContent=Xe),fe=o(t),x=s(t,"H3",{class:!0,"data-svelte-h":!0}),r(x)!=="svelte-sd78b9"&&(x.textContent=Ke),ge=o(t),W=s(t,"P",{"data-svelte-h":!0}),r(W)!=="svelte-dlpp8b"&&(W.textContent=Ze),me=o(t),k=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r(k)!=="svelte-1gunb0x"&&(k.innerHTML=et),ve=o(t),V=s(t,"P",{"data-svelte-h":!0}),r(V)!=="svelte-1s28fbu"&&(V.textContent=tt),be=o(t),z=s(t,"P",{"data-svelte-h":!0}),r(z)!=="svelte-19e92er"&&(z.innerHTML=nt),_e=o(t),R=s(t,"P",{"data-svelte-h":!0}),r(R)!=="svelte-f74p8c"&&(R.innerHTML=at),ye=o(t),ne(w.$$.fragment,t),xe=o(t),S=s(t,"P",{"data-svelte-h":!0}),r(S)!=="svelte-15wp5eu"&&(S.textContent=ot),ke=o(t),ne(L.$$.fragment,t),we=o(t),U=s(t,"P",{"data-svelte-h":!0}),r(U)!=="svelte-1r0454s"&&(U.textContent=it),Le=o(t),ne(C.$$.fragment,t),Ce=o(t),F=s(t,"P",{"data-svelte-h":!0}),r(F)!=="svelte-1yk881q"&&(F.innerHTML=st),Te=o(t),O=s(t,"P",{"data-svelte-h":!0}),r(O)!=="svelte-hpxw1r"&&(O.textContent=rt),Me=o(t),E=s(t,"P",{"data-svelte-h":!0}),r(E)!=="svelte-1nnd1vi"&&(E.textContent=lt),Ie=o(t),T=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r(T)!=="svelte-1xkci4m"&&(T.innerHTML=ct),He=o(t),J=s(t,"P",{"data-svelte-h":!0}),r(J)!=="svelte-16v4mtd"&&(J.textContent=dt),qe=o(t),M=s(t,"H3",{class:!0,"data-svelte-h":!0}),r(M)!=="svelte-1iab2tq"&&(M.textContent=ut),$e=o(t),B=s(t,"P",{"data-svelte-h":!0}),r(B)!=="svelte-1r592lv"&&(B.innerHTML=ht),Ae=o(t),G=s(t,"P",{"data-svelte-h":!0}),r(G)!=="svelte-1jt1jcf"&&(G.textContent=pt),Pe=o(t),N=s(t,"P",{"data-svelte-h":!0}),r(N)!=="svelte-vpxgaw"&&(N.innerHTML=ft),je=o(t),ne(I.$$.fragment,t),De=o(t),Y=s(t,"P",{"data-svelte-h":!0}),r(Y)!=="svelte-10n91gx"&&(Y.textContent=gt),We=o(t),H=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r(H)!=="svelte-1eusgyu"&&(H.innerHTML=mt),Ve=o(t),g=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r(g)!=="svelte-1qx4u3d"&&(g.innerHTML=vt),ze=o(t),q=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r(q)!=="svelte-1a4j683"&&(q.innerHTML=bt),Re=o(t),m=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r(m)!=="svelte-1lxoetg"&&(m.innerHTML=_t),Se=o(t),$=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r($)!=="svelte-1gjahab"&&($.innerHTML=yt),Ue=o(t),v=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r(v)!=="svelte-xq0uqn"&&(v.innerHTML=xt),Fe=o(t),A=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r(A)!=="svelte-18hsmgg"&&(A.innerHTML=kt),Oe=o(t),b=s(t,"DIV",{class:!0,"data-svelte-h":!0}),r(b)!=="svelte-1of0xvm"&&(b.innerHTML=wt),Ee=o(t),Q=s(t,"P",{"data-svelte-h":!0}),r(Q)!=="svelte-xmpcd9"&&(Q.textContent=Lt),t.forEach(Be),ee.forEach(Be),this.h()},h(){l(h,"class","h1 text-center mb-12"),l(p,"class","h2"),l(_,"class","card variant-filled-ghost p-4"),l(y,"class","h3"),l(x,"class","h3"),l(k,"class","flex justify-center mt-0"),l(T,"class","container h-full mx-auto flex justify-center items-center leading-relaxed"),l(M,"class","h3"),l(H,"class","text-center"),l(g,"class","flex justify-center mt-0"),l(q,"class","text-center"),l(m,"class","flex justify-center mt-0"),l($,"class","text-center"),l(v,"class","flex justify-center mt-0"),l(A,"class","text-center"),l(b,"class","flex justify-center mt-0"),l(e,"class","space-y-5 m-10 custom-container svelte-1bdawuf"),l(d,"class","container h-full mx-auto flex justify-center items-center leading-relaxed")},m(c,ee){At(c,d,ee),n(d,e),n(e,h),n(e,K),n(e,p),n(e,P),n(e,f),n(e,de),n(e,_),n(e,ue),n(e,y),n(e,he),n(e,j),n(e,pe),n(e,D),n(e,fe),n(e,x),n(e,ge),n(e,W),n(e,me),n(e,k),n(e,ve),n(e,V),n(e,be),n(e,z),n(e,_e),n(e,R),n(e,ye),ae(w,e,null),n(e,xe),n(e,S),n(e,ke),ae(L,e,null),n(e,we),n(e,U),n(e,Le),ae(C,e,null),n(e,Ce),n(e,F),n(e,Te),n(e,O),n(e,Me),n(e,E),n(e,Ie),n(e,T),n(e,He),n(e,J),n(e,qe),n(e,M),n(e,$e),n(e,B),n(e,Ae),n(e,G),n(e,Pe),n(e,N),n(e,je),ae(I,e,null),n(e,De),n(e,Y),n(e,We),n(e,H),n(e,Ve),n(e,g),n(e,ze),n(e,q),n(e,Re),n(e,m),n(e,Se),n(e,$),n(e,Ue),n(e,v),n(e,Fe),n(e,A),n(e,Oe),n(e,b),n(e,Ee),n(e,Q),Z=!0,Je||(Ct=[X(h,"click",u[1]),X(g,"click",u[2]),X(m,"click",u[3]),X(v,"click",u[4]),X(b,"click",u[5])],Je=!0)},p:Pt,i(c){Z||(oe(w.$$.fragment,c),oe(L.$$.fragment,c),oe(C.$$.fragment,c),oe(I.$$.fragment,c),Z=!0)},o(c){ie(w.$$.fragment,c),ie(L.$$.fragment,c),ie(C.$$.fragment,c),ie(I.$$.fragment,c),Z=!1},d(c){c&&Be(d),se(w),se(L),se(C),se(I),Je=!1,jt(Ct)}}}function St(u){window.location.href=u}function Ut(u){let d=Vt();function e(P){console.log("image",P);const f={image:P,modalClasses:"max-w-[90%] max-h-[90%] rounded-container-token overflow-hidden shadow-xl"};d.trigger(f)}return[e,()=>St("/"),()=>e(Mt),()=>e(It),()=>e(Ht),()=>e(qt)]}class Gt extends Dt{constructor(d){super(),Wt(this,d,Ut,Rt,$t,{})}}export{Gt as component};
