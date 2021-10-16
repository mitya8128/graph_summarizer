### Graph_summarizer   
 &nbsp;

Summarize long texts by combining graph-algorithmic approaches with distributive language vector models.   

Work in progress.   

Some code was taken from my [another](https://github.com/mitya8128/nlp_graph) repository.   
&nbsp; 
### Main idea:  
- Extract important words by [wordrank algorithm](https://github.com/mitya8128/graph_summarizer/blob/master/wordrank.py) for each sentences of text 
  - word2vec based word vectorization 
  - build adjacency matrix on top of that (distance between words in sentence)
  - transfer adjacency matrix into weighted graph
  - find clique of graph with max length
  - therefore we deduce the most "important" words in graph-theoretic means
- Extract important sentences of text through [textrank algorithm](https://github.com/mitya8128/graph_summarizer/blob/master/textrank_sentence.py) - function [*build_similarity_matrix*](https://github.com/mitya8128/graph_summarizer/blob/472d70752a572fe7cb71272935072a0149b914b1/textrank_sentence.py#L55) as an entrypoint 
  - basically it compares sentences and deduce metric of similarity based on simple equality of tokens from sentences
  - thus with that information we could find the most "informative" sentences (by means of high similarity metric)
- If necessary you could run whole algorithm several times through text (could be useful if you want to compress text more) - function [*generate_summary_loop*](https://github.com/mitya8128/graph_summarizer/blob/472d70752a572fe7cb71272935072a0149b914b1/textrank_sentence.py#L111) as a full description of pipeline
 &nbsp;  
### Demos:      
[demo_russian](https://github.com/mitya8128/graph_summarizer/blob/master/demo_russian.ipynb)   
 &nbsp;  
