## ALTERNATIVE QUERY GENERATION STRATEGIES

As stated in Section 4.1 of the work, we conducted preliminary experiments with traditional and newer simulation approaches, including methods oriented to sample queries from classic language models and neural techniques based on docT5query and keyBERT. However, they tended to produce unrealistic queries drifting away from the topic of the source article. Thus, we decided to use a Generative-AI approach powered by Llama3.

### PRELIMINARY EXPERIMENTS

- doc2query.py: this script contains the preliminary generation experiments with doc2query model.
- keyterms.py: this script contains the query generation process with KeyBERT, sampling and concatenating top k terms from the documents.

To evaluate them, we computed their semantic similarity to the original queries. Both methods yielded **inferior similarity** to the one obtained with GenAI. Moreover, we **manually inspected** the queries and observed that in some cases they drifted away from the main topic (e.g.*pelosi mccarthy bipartisan filibustered senate* for a news accusing Nancy Pelosi of a security breakdown in Jan. 6th events or *why did Paul Gosar ask for a red flag* focused only in the main character that accused Biden of corruption).