 Large language models (LLMs) have shown impressive capabilities in specialized domains like
 mathematics and coding, yet often rely on surface-level patterns rather than genuine systematic
 reasoning. This thesis addresses an underexplored area in current benchmarks: how LLMs handle
 fundamental cognitive skills like object tracking and spatial reasoning. This work introduces a
 novel benchmark with four distinct reasoning categories (Basic Swap, Negations, Overlap, and
 Rules).
 
 The benchmark includes a core dataset (1,200 questions) and an extended version with elon
gated contexts to test robustness. Five small instruction-tuned LLMs (DeepSeek-R1-Distill-Qwen
1.5B, Llama-3.2-1B/3B, Phi-3.5-Mini, and Qwen2.5-1.5B) were evaluated under three prompt
ing strategies: providing examples with few-shot, step-by-step reasoning prompts with chain-of
thought (CoT), and no CoT (direct answering).

 The results reveal limitations in tested models: all models perform near random guessing
 (33.3%) on Negations and Rules tasks, with accuracy dropping further in extended contexts. For
 example, even top-performing models like Phi-3.5-Mini saw accuracy fall from 89% to 31% when
 context length increased. DeepSeek, in particular, completed only 23.1% of CoT responses in ex
tended settings, highlighting fundamental processing limitations. While CoT prompting improved
 accuracy for most normal-length tasks, it often degraded performance in extended contexts due
 to overthinking. Additionally, models exhibited strong answer distribution biases, such as Llama
3.2-1B’s marked preference for option B, which amplified with longer contexts, while DeepSeek
 maintained even answer distributions across different context lengths, suggesting greater resis
tance to positional bias that favors specific answer options regardless of content. Architectural
 differences were also apparent, with models excelling at different tasks. For instance, Phi-3.5-Mini
 performed best on Overlap tasks (89% accuracy), while DeepSeek showed strong results on Basic
 Swap tasks (65% accuracy).
 
 These findings demonstrate that reasoning limitations persist even in newer models, replicating
 error types such as misreading, logical flaws, and generating incorrect details (hallucination) iden
tified in prior research. Overall, this work underscores the need for benchmarks that evaluate not
 just final answers but the reasoning process itself. These results highlight fundamental reasoning
 limitations in current LLMs and identify specific challenges in spatial reasoning that persist despite
 advances in other areas.
