# Standard RAG vs Agentic RAG

## What is Retrieval-Augmented Generation?
Retrieval-Augmented Generation (RAG) combines a retriever with a large language model. Instead of relying only on what the model memorised during training, the system first retrieves relevant passages from an external knowledge base and then asks the model to answer using those passages as context. This reduces hallucinations, keeps answers up to date, and makes it possible to cite sources.

## The Standard RAG pipeline
A standard RAG system follows a fixed, linear pipeline: the user question is embedded, the top-k most similar chunks are retrieved from a vector database, and the chunks are pasted into a prompt together with the question. The model generates an answer in a single pass. Standard RAG always retrieves exactly once, with the original query, regardless of whether retrieval is needed or whether the results are good.

Main limitations of standard RAG:
- It cannot decompose multi-part questions into sub-questions.
- A poorly phrased query leads to poor retrieval, and there is no second attempt.
- It cannot combine retrieval with other tools such as a calculator or data analysis.
- It retrieves even for greetings or questions that need no documents.

## What makes RAG "agentic"?
Agentic RAG puts an LLM-based agent in control of the retrieval process. The agent first plans: it classifies the question, decides whether retrieval is needed, and breaks complex questions into steps. It then chooses a retrieval strategy (semantic, keyword or hybrid search, or reading a whole document), evaluates the retrieved evidence, and refines the query if the results are weak. The agent can call other tools such as summarisation, comparison, data analysis and calculation, and finally composes an answer grounded in the evidence, with citations.

Key capabilities of Agentic RAG:
1. Planning before answering.
2. Dynamic decision of when and how to retrieve.
3. Query refinement and multi-step retrieval (multi-hop reasoning).
4. Tool use beyond search.
5. Self-reflection: checking whether the evidence actually answers the question.

## Trade-offs
Agentic RAG produces higher-quality answers on complex questions, but it is slower and more expensive because it makes several model and tool calls per question. For simple factual lookups, standard RAG is often sufficient. A practical design routes simple questions to a fast path and lets the agent handle complex ones.

## Evaluation
Common RAG evaluation metrics include context precision (are the retrieved chunks relevant?), context recall (was all the needed information retrieved?), faithfulness (is the answer supported by the context?) and answer relevance. Agentic systems should additionally be evaluated on tool-selection accuracy and the number of steps taken.
