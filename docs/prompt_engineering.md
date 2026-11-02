# Prompt Engineering

Prompt engineering is the practice of writing instructions that get reliable, useful output from a language model. Small changes in wording can produce very different results.

A system prompt sets the rules and role for the model before the conversation begins. In a RAG application, the system prompt typically tells the model to answer only from the provided context, to cite sources, and to admit when it does not know.

Good prompts are specific. Instead of asking "Summarize this," a better prompt says "Summarize this article in three bullet points for a beginner, focusing on the main takeaway." Specific prompts reduce guessing.

Giving examples in the prompt is called few-shot prompting. Showing the model one or two examples of the desired input and output format often improves consistency more than describing the format in words.

Prompt injection is an attack where a user or a document includes instructions meant to override the system prompt, such as "Ignore all previous instructions." Defenses include validating input, separating instructions from data clearly, and checking the model's output before showing it.