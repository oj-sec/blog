import{s as Ne,e as i,a,c as s,b as Ke,g as l,d as o,f as _e,h as c,B as Qe,i as Xe,j as n,l as Ze,n as et}from"../chunks/scheduler.DzZHELrM.js";import{S as tt,i as nt,c as z,a as D,m as J,t as O,b as V,d as F}from"../chunks/index.BWQOah18.js";import"../chunks/ProgressBar.svelte_svelte_type_style_lang.BrI0fzGN.js";import{C as Y}from"../chunks/CodeBlock.Cigd8BMN.js";const at=""+new URL("../assets/predictions.Blml3tS9.png",import.meta.url).href;function ot(S){let h,e,u,ke="@oj-sec",K,d,Pe=`Evaluating Large Language Models as future event forecasters - Part
            One: Introduction`,N,_,Le="4 April 2024",Q,p,Te=`You can access a Juypter notebook (built for Colab) associated with
            this post <a href="https://github.com/oj-sec/blog/blob/main/notebooks/20240404.ipynb" style="text-decoration: underline; color: lightblue;">here</a>.`,X,m,$e="The pitch",Z,k,Ie=`Forecasting future events is at the heart of intelligence analysis.
            In cyber threat intelligence, we are in particular danger of losing
            sight of the future due to the centrality of the past in intrusion
            analysis, our lifeblood and primary collection source. But the past
            can only be valuable to decision makers when it informs actions to
            avoid or mitigate future threats. If we cannot extrapolate forecasts
            into a model of the future, we are ultimately failing to realise the
            promise of threat intelligence and will inevitably underserve
            consumers.`,ee,P,Me=`The largest systematic evaluation of forecasting performance is the
            Aggregative Contingent Estimation (ACE) program run by the US
            Intelligence Advanced Research Projects Activity (IARPA) from 2010
            to 2015. The ACE pitted research projects against one another
            including advanced market and algorithm-based solutions and
            professional intelligence analysts with access to classified
            information.`,te,L,je=`The ACE was won by a system that crowdsourced opinion from
            laypeople, Philip Tetlock's Good Judgement Project. In his various
            retrospectives, Tetlock suggests that the performance of the Good
            Judgement Project is a combination of the wisdom of the crowd and
            the ability of some individuals to forecast particularly effectively
            due to a particular collection of personality traits and an
            effective approach to research and information synthesis.`,ne,T,Ae=`In this blog series, I'm going to investigate the capability and
            applicability of Large Language Models (LLMs) for predicting future
            events in an intelligence analysis context. In theory, there is
            scope for a LLM consuming text-based information about current
            events to emulate both the wisdom of the crowd and the analytical
            processes and dispositions of Tetlock's superforecasters. While the
            effectiveness of such a system is an open question, I'll use it as a
            lense to explore some LLM concepts and design patterns that should
            salvage some value from even a (perhaps likely) complete failure.`,ae,$,qe=`I expect to cover LLM behavior and performance control techniques
            including:`,oe,f,Ge="<li>- restricting output tokens</li> <li>- token sampling</li> <li>- prompt engineering techniques</li> <li>- agent patterns</li> <li>- retrieval augmented generation</li> <li>- parameter-efficient fine tuning</li>",ie,g,We="Background",se,I,He=`We typically interact with LLMs served to us via a remote API in a
            chat-like format. When we interact with a system like OpenAI's
            ChatGPT, inference against the model occurs on a very powerful cloud
            server in a data center. But it's possible for us to run models on
            our own hardware - usually called a "local" model - allowing for
            fine-grained control over a model's behaviour. In particular, we can
            restrict a LLM's generation so that it matches a specific format.
            This can be an essential element of programmatically handling the
            outputs of LLMs and developing applications that go beyond simple
            chats with human users.`,le,M,Ue=`For example, if I give ChatGPT 3.5 an ACE-style prompt relating to
            an event with major geopolitical significance through 2025:`,re,v,ce,j,Ee="I receive a verbose response beginning with:",he,w,ue,A,Se=`The response does not contain a numerical prediction at all. We have
            at least two major problems here - ChatGPT has a tendency to be
            verbose and wrap its answers in explanatory text, and ChatGPT has
            safety training that causes it to be less equivocal and more guarded
            about certain topics, including politics. While we could attempt to
            parse ChatGPT's answers into what we want or use prompt engineering
            to encourage better adherence to the task, these approaches have
            strong diminishing returns and aren't universally reliable.
            Moreover, we simply don't need to engage in imprecise methods when
            using a model we have direct control over.`,de,q,Be=`We can exert direct control over a local LLM using Llama.cpp invoked
            via Microsoft&#39;s Guidance library. LLama.cpp lets us run inference
            using flexible hardware (including splitting the model between
            CPU+RAM and GPU+VRAM) and Guidance gives us controlled outputs with
            straightforward syntax. We can replicate our above experiment under
            Guidance with the following Python code. You can find a Jupyter
            notebook showing the installation process in the repository
            associated with the blog series <a href="https://github.com/oj-sec/blog/blob/main/notebooks/20240404.ipynb" style="text-decoration: underline; color: lightblue;">here</a>.`,pe,b,me,G,Re="Which gives us back a result consistent with the constraint we made:",fe,x,ge,W,ze=`We can confidently parse the output of the model into a number and
            use it in a subsequent data analysis pipeline. We have effectively
            solved sensitive topic avoidance and cut through the waffle in a
            single step that also saved us wasted time generating irrelevant
            material.`,ve,y,De="A proof of concept",we,H,Je=`To finish off, let's demonstrate that the forecasts we can elicit
            from a LLM have some sort of underlying logic. To correspond with
            our existing prompt, which relates to a scenario that has a real
            chance of actually occurring, let's present a scenario that (pending
            constitutional change) cannot occur: "A horse will win the 2024 US
            election." If we sample our language model 100 times on both
            questions, we get the following distribution of predictions:`,be,C,Oe,xe,U,Ve=`We can see a somewhat untidy normal distribution for the Trump 2024
            prediction, centered around the 0.20-0.30 likelihood range. The
            distribution of the data indicates that there is a weak consensus
            within the model that Trump is unlikely to win the 2024 US election.
            We can also observe that there is an absolute consensus within the
            model that there is no chance that a horse will win the 2024 US
            election. These distributions suggest that there is some cogency to
            the models' predictions and that it is drawing on contextual
            training data. We can regard this as proof that the underlying idea
            is not inherently infeasible - its just a question of how far we can
            push the accuracy.`,ye,E,Fe=`But it may be quite a large and difficult question. The Trump
            prediction is lower than what we might expect given that bookmakers
            give the scenario approximately even odds of occurring at the time
            of writing. If we look at the individual predictions made by the
            model, we can see that all 100 predictions were evenly divisible by
            0.1. This strong bias towards round numbers will almost certainly
            hurt us as we take this project forward - one of Tetlock's
            observations about high performing forecasters was a tendency
            towards oddly precise numeric predictions and adjustments on the
            scale of single percentages. These shortfalls are likely to be some
            combination of training on old data, meaning no access to
            contemporary information, poor analytical reasoning, particularly a
            poor ability to model the world, and bias in one direction or the
            other. Attempting to address these issues will be the subject of
            future posts in this series.`,B,Ce,Ye;return v=new Y({props:{language:"plaintext",code:`Predict the likelihood of the following outcome on a scale from 0.00 to 1.00, with 0.00 meaning the event is impossible and 1.00 meaning the event is certain to occur:
"Donald Trump will win the 2024 US election"`}}),w=new Y({props:{language:"plaintext",code:`As an AI language model, I don't have access to real-time data or the ability to predict specific future events accurately. However, I can provide an analysis based on the information available up to my last training data in January 2022.

Predicting the outcome of a future election involves numerous factors including political dynamics, public opinion, candidate performance, and unforeseen events. Given the volatility of politics and the uncertainty surrounding future events, it's challenging to assign a precise likelihood to any specific outcome.`}}),b=new Y({props:{language:"python",code:`from guidance import models, gen
    
# load a local model downloaded to our system
llm = models.LlamaCpp("./models/mistral-7b-openorca.Q4_K_M.gguf", n_gpu_layers=20, n_ctx=1000)

# define a regular expression to match the numbers between 0.00 and 1.00 inclusive
output_regex = r"(0\\.\\d\\d|1\\.00)"

# define our prompt
prompt = 'Predict the likelihood of the following outcome on a scale from 0.00 to 1.00, with 0.00 meaning the event is impossible and 1.00 meaning the event is certain to occur: "Donald Trump will win the 2024 US election."\\nPREDICTION:'

# run constrained inference 
output = llm + prompt + gen(name="response", regex=output_regex, max_tokens=20, temperature=0.7)
print(output['response'])`}}),x=new Y({props:{language:"plaintext",code:"0.20"}}),{c(){h=i("div"),e=i("div"),u=i("h1"),u.textContent=ke,K=a(),d=i("h2"),d.textContent=Pe,N=a(),_=i("p"),_.textContent=Le,Q=a(),p=i("p"),p.innerHTML=Te,X=a(),m=i("h3"),m.textContent=$e,Z=a(),k=i("p"),k.textContent=Ie,ee=a(),P=i("p"),P.textContent=Me,te=a(),L=i("p"),L.textContent=je,ne=a(),T=i("p"),T.textContent=Ae,ae=a(),$=i("p"),$.textContent=qe,oe=a(),f=i("ul"),f.innerHTML=Ge,ie=a(),g=i("h3"),g.textContent=We,se=a(),I=i("p"),I.textContent=He,le=a(),M=i("p"),M.textContent=Ue,re=a(),z(v.$$.fragment),ce=a(),j=i("p"),j.textContent=Ee,he=a(),z(w.$$.fragment),ue=a(),A=i("p"),A.textContent=Se,de=a(),q=i("p"),q.innerHTML=Be,pe=a(),z(b.$$.fragment),me=a(),G=i("p"),G.textContent=Re,fe=a(),z(x.$$.fragment),ge=a(),W=i("p"),W.textContent=ze,ve=a(),y=i("h3"),y.textContent=De,we=a(),H=i("p"),H.textContent=Je,be=a(),C=i("img"),xe=a(),U=i("p"),U.textContent=Ve,ye=a(),E=i("p"),E.textContent=Fe,this.h()},l(r){h=s(r,"DIV",{class:!0});var R=Ke(h);e=s(R,"DIV",{class:!0});var t=Ke(e);u=s(t,"H1",{class:!0,"data-svelte-h":!0}),l(u)!=="svelte-aomebi"&&(u.textContent=ke),K=o(t),d=s(t,"H2",{class:!0,"data-svelte-h":!0}),l(d)!=="svelte-184jdd4"&&(d.textContent=Pe),N=o(t),_=s(t,"P",{"data-svelte-h":!0}),l(_)!=="svelte-1hmt4gw"&&(_.textContent=Le),Q=o(t),p=s(t,"P",{class:!0,"data-svelte-h":!0}),l(p)!=="svelte-105mgyz"&&(p.innerHTML=Te),X=o(t),m=s(t,"H3",{class:!0,"data-svelte-h":!0}),l(m)!=="svelte-e3czwl"&&(m.textContent=$e),Z=o(t),k=s(t,"P",{"data-svelte-h":!0}),l(k)!=="svelte-15wz9gh"&&(k.textContent=Ie),ee=o(t),P=s(t,"P",{"data-svelte-h":!0}),l(P)!=="svelte-1vflgq0"&&(P.textContent=Me),te=o(t),L=s(t,"P",{"data-svelte-h":!0}),l(L)!=="svelte-5pluwb"&&(L.textContent=je),ne=o(t),T=s(t,"P",{"data-svelte-h":!0}),l(T)!=="svelte-d86ayj"&&(T.textContent=Ae),ae=o(t),$=s(t,"P",{"data-svelte-h":!0}),l($)!=="svelte-1iyhbln"&&($.textContent=qe),oe=o(t),f=s(t,"UL",{class:!0,"data-svelte-h":!0}),l(f)!=="svelte-19ukkpq"&&(f.innerHTML=Ge),ie=o(t),g=s(t,"H3",{class:!0,"data-svelte-h":!0}),l(g)!=="svelte-14920e"&&(g.textContent=We),se=o(t),I=s(t,"P",{"data-svelte-h":!0}),l(I)!=="svelte-1d9e41u"&&(I.textContent=He),le=o(t),M=s(t,"P",{"data-svelte-h":!0}),l(M)!=="svelte-1xl1f1l"&&(M.textContent=Ue),re=o(t),D(v.$$.fragment,t),ce=o(t),j=s(t,"P",{"data-svelte-h":!0}),l(j)!=="svelte-13qe5n9"&&(j.textContent=Ee),he=o(t),D(w.$$.fragment,t),ue=o(t),A=s(t,"P",{"data-svelte-h":!0}),l(A)!=="svelte-38nchj"&&(A.textContent=Se),de=o(t),q=s(t,"P",{"data-svelte-h":!0}),l(q)!=="svelte-17jzlxz"&&(q.innerHTML=Be),pe=o(t),D(b.$$.fragment,t),me=o(t),G=s(t,"P",{"data-svelte-h":!0}),l(G)!=="svelte-a6j11v"&&(G.textContent=Re),fe=o(t),D(x.$$.fragment,t),ge=o(t),W=s(t,"P",{"data-svelte-h":!0}),l(W)!=="svelte-1ym6m6c"&&(W.textContent=ze),ve=o(t),y=s(t,"H3",{class:!0,"data-svelte-h":!0}),l(y)!=="svelte-1fit1v8"&&(y.textContent=De),we=o(t),H=s(t,"P",{"data-svelte-h":!0}),l(H)!=="svelte-19roxj0"&&(H.textContent=Je),be=o(t),C=s(t,"IMG",{src:!0,alt:!0,class:!0}),xe=o(t),U=s(t,"P",{"data-svelte-h":!0}),l(U)!=="svelte-1t1r9v8"&&(U.textContent=Ve),ye=o(t),E=s(t,"P",{"data-svelte-h":!0}),l(E)!=="svelte-ecn7v2"&&(E.textContent=Fe),t.forEach(_e),R.forEach(_e),this.h()},h(){c(u,"class","h1 text-center mb-12"),c(d,"class","h2"),c(p,"class","card variant-filled-ghost p-4"),c(m,"class","h3"),c(f,"class","ml-8"),c(g,"class","h3"),c(y,"class","h3"),Qe(C.src,Oe=at)||c(C,"src",Oe),c(C,"alt","Predictions"),c(C,"class","w-2/1 min-w-1200 mx-auto"),c(e,"class","space-y-5 m-10 custom-container svelte-1bmufab"),c(h,"class","container h-full mx-auto flex justify-center items-center leading-relaxed")},m(r,R){Xe(r,h,R),n(h,e),n(e,u),n(e,K),n(e,d),n(e,N),n(e,_),n(e,Q),n(e,p),n(e,X),n(e,m),n(e,Z),n(e,k),n(e,ee),n(e,P),n(e,te),n(e,L),n(e,ne),n(e,T),n(e,ae),n(e,$),n(e,oe),n(e,f),n(e,ie),n(e,g),n(e,se),n(e,I),n(e,le),n(e,M),n(e,re),J(v,e,null),n(e,ce),n(e,j),n(e,he),J(w,e,null),n(e,ue),n(e,A),n(e,de),n(e,q),n(e,pe),J(b,e,null),n(e,me),n(e,G),n(e,fe),J(x,e,null),n(e,ge),n(e,W),n(e,ve),n(e,y),n(e,we),n(e,H),n(e,be),n(e,C),n(e,xe),n(e,U),n(e,ye),n(e,E),B=!0,Ce||(Ye=Ze(u,"click",S[0]),Ce=!0)},p:et,i(r){B||(O(v.$$.fragment,r),O(w.$$.fragment,r),O(b.$$.fragment,r),O(x.$$.fragment,r),B=!0)},o(r){V(v.$$.fragment,r),V(w.$$.fragment,r),V(b.$$.fragment,r),V(x.$$.fragment,r),B=!1},d(r){r&&_e(h),F(v),F(w),F(b),F(x),Ce=!1,Ye()}}}function it(S){window.location.href=S}function st(S){return[()=>it("/")]}class ut extends tt{constructor(h){super(),nt(this,h,st,ot,Ne,{})}}export{ut as component};
