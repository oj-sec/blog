<script>
    import predictions from './predictions.png';
    import { CodeBlock } from "@skeletonlabs/skeleton";


    function navigate(url) {
        window.location.href = url;
    }

</script>

<div class="container h-full mx-auto flex justify-center items-center w-1/2">
	<div class="space-y-5 m-10">
		<h1 class="h1 text-center mb-12" on:click={() => navigate("/")} >@oj-sec</h1>

        <h2 class="h2">Evaluating Large Language Models as future event forecasters - Part One</h2>
        <p>4 April 2024</p>
        <h3 class="h3">The pitch</h3>
        <p>Forecasting future events is at the heart of intelligence analysis. In cyber threat intelligence, we are in particular danger of losing sight of the future due to the centrality of the past in intrusion analysis, our lifeblood and primary collection source. But the past can only be valuable to decision makers when it informs actions to avoid or mitigate future threats. If we cannot extrapolate forecasts into a model of the future, we are ultimately failing to realise the promise of threat intelligence and will inevitably underserve consumers. 
        </p>
        <p>The largest systematic evaluation of forecasting performance is the Aggregative Contingent Estimation (ACE) program run by the US Intelligence Advanced Research Projects Activity (IARPA) from 2010 to 2015. The ACE pitted research projects against one another including advanced market and algorithm-based solutions and professional intelligence analysts with access to classified information.  
        </p>
        <p>The ACE was won by a system that crowdsourced opinion from laypeople, Philip Tetlock's Good Judgement Project. In his various retrospectives, Tetlock suggests that the performance of the Good Judgement Project is a combination of the wisdom of the crowd and the ability of some individuals to forecast particularly effectively due to a particular collection of personality traits and an effective approach to research and information synthesis.</p>
        <p>In this blog series, I'm going to investigate the capability and applicability of Large Language Models (LLMs) for predicting future events in an intelligence analysis context. In theory, there is scope for a LLM consuming text-based information about current events to emulate both the wisdom of the crowd and the analytical processes and dispositions of Tetlock's superforecasters. While the effectiveness of such a system is an open question, I'll use it as a lense to explore some LLM concepts and design patterns that should salvage some value from even a (perhaps likely) complete failure. 
        </p>
        <p>
        I expect to cover LLM behavior and performance control techniques including:
        <ul class="ml-8">
            <li>- restricting output tokens</li>
            <li>- token sampling</li>
            <li>- prompt engineering techniques</li>
            <li>- agent patterns</li>
            <li>- retrieval augmented generation</li>
            <li>- parameter-efficient fine tuning</li>
        </ul>
        <h3 class="h3">Background</h3>
        <p>We typically interact with LLMs served to us via a remote API in a chat-like format. When we interact with a system like OpenAI's ChatGPT, inference against the model occurs on a very powerful cloud server in a data center. But it's possible for us to run models on our own hardware - usually called a "local" model - allowing for fine-grained control over a model's behaviour. In particular, we can restrict a LLM's generation so that it matches a specific format. This can be an essential element of programmatically handling the outputs of LLMs and developing applications that go beyond simple chats with human users. 
        <p>For example, if I give ChatGPT 3.5 an ACE-style prompt:</p>
        
        <CodeBlock language="plaintext" code={`Predict the likelihood of the following outcome on a scale from 0.00 to 1.00, with 0.00 meaning the event is impossible and 1.00 meaning the event is certain to occur:
"Donald Trump will win the 2024 US election"`}></CodeBlock>
        <p>I receive a verbose response beginning with: 
        </p>
        <CodeBlock language="plaintext" code={`As an AI language model, I don't have access to real-time data or the ability to predict specific future events accurately. However, I can provide an analysis based on the information available up to my last training data in January 2022.

Predicting the outcome of a future election involves numerous factors including political dynamics, public opinion, candidate performance, and unforeseen events. Given the volatility of politics and the uncertainty surrounding future events, it's challenging to assign a precise likelihood to any specific outcome.`}></CodeBlock>
    <p>The response does not contain a numerical prediction at all. We have at least two major problems here - ChatGPT has a tendency to be verbose and wrap its answers in explanatory text, and ChatGPT has safety training that causes it to be less equivocal and more guarded about certain topics, including  politics. While we could attempt to parse ChatGPT's answers into what we want or use prompt engineering to encourage better adherence to the task, these approaches have strong diminishing returns and aren't universally reliable. Moreover, we simply don't need to engage in imprecise methods when using a model we have direct control over. 
    </p>
    <p>We can exert direct control over a local LLM using Llama.cpp invoked via Microsoft's Guidance library. LLama.cpp lets us run inference using flexible hardware (including splitting the model between CPU+RAM and GPU+VRAM) and Guidance gives us controlled outputs with straightforward syntax. We can replicate our above experiment under Guidance with the following Python code. You can find a Jupyter notebook showing the installation process in the repository associated with the blog series <a href="https://github.com/oj-sec/blog/blob/main/notebooks/20240404" style="text-decoration: underline; color: lightblue;">here</a>.
    </p>
    <CodeBlock language="python" code={`from guidance import models, gen
    
# load a local model downloaded to our system
llm = models.LlamaCpp("./models/mistral-7b-openorca.Q4_K_M.gguf", n_gpu_layers=20, n_ctx=1000)

# define a regular expression to match the numbers between 0.00 and 1.00 inclusive
output_regex = r"(0\\.\\d\\d|1\\.00)"

# define our prompt
prompt = 'Predict the likelihood of the following outcome on a scale from 0.00 to 1.00, with 0.00 meaning the event is impossible and 1.00 meaning the event is certain to occur: "Donald Trump will win the 2024 US election."\\nPREDICTION:'

# run constrained inference 
output = llm + prompt + gen(name="response", regex=output_regex, max_tokens=20, temperature=0.7)
print(output['response'])`}></CodeBlock>
<p>Which gives us back a result consistent with the constraint we made:</p>
<CodeBlock language="plaintext" code={`0.20`}></CodeBlock>
<p>We can confidently parse the output of the model into a number and use it in a subsequent data analysis pipeline. We have effectively solved sensitive topic avoidance and cut through the waffle in a single step that also saved us wasted time generating irrelevant material. 
</p>
<h3 class="h3">A proof of concept </h3>
<p>To finish off, let's demonstrate that the forecasts we can elicit from a LLM have some sort of underlying logic. To correspond with our existing prompt, which relates to a scenario that has a real chance of actually occurring, let's present a scenario that (pending constitutional change) cannot occur: "A horse will win the 2024 US election." If we sample our language model 100 times on both questions, we get the following distribution of predictions:
</p>
<img src={predictions} alt="Predictions" class="w-2/1 min-w-1200 mx-auto">
<p>We can see a relatively shapely normal distribution for the Trump 2024 prediction, centered around the 0.20-0.30 likelihood range. The distribution of the data indicates that there is a weak consensus within the model that Trump is unlikely to win the 2024 US election. We can also observe that there is an absolute consensus within the model that there is no chance that a horse will win the 2024 US election. These distributions suggest that there is some cogency to the models' predictions and that it is drawing on contextual training data. We can regard this as proof that the underlying idea is not inherently infeasible - its just a question of how far we can push the accuracy.
</p>
<p>But it may be quite a large and difficult question. The Trump prediction is lower than what we might expect given that bookmakers give the scenario approximately even odds of occurring at the time of writing. If we look at the individual predictions made by the model, we can see that all 100 predictions were evenly divisible by 0.1. This strong bias towards round numbers will almost certainly hurt us as we take this project forward - one of Tetlock's observations about high performing forecasters was a tendency towards oddly precise numeric predictions and adjustments on the scale of single percentages. These shortfalls are likely to be some combination of training on old data, meaning no access to contemporary information, poor analytical reasoning, particularly a poor ability to model the world, and bias in one direction or the other. Attempting to address these issues will be the subject of future posts in this series. </p>
</div>
</div>