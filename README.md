# Labs and Homeworks for Information Retrieval 1 course, MSc AI @ UvA 2017/2018.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  
Solutions and implementation from [Davide Belli](https://github.com/davide-belli), [Gabriele Cesa](https://github.com/Gabri95) and [Linda Petrini](https://github.com/LindaPetrini).
---

## Lab 1 - Evaluation Measures, Interleaving and Click Models
#### Topics:
Commercial search engines typically use a funnel approach in evaluating a new search algorithm: they first use an offline test collection to compare the production algorithm (P) with the new experimental algorithm (E); if E outperforms P with respect to the evaluation measure of their interest, the two algorithms are then compared online through an interleaving experiment.

Therefore, one of the key questions in IR is whether offline evaluation and online evaluation outcomes agree with each other. In this lab we run experiments in which we compare different online and offline evaluation metrics in order to understand the relationship between them. Multiple click models are used in order to simulate user clicks.

## Lab 2 - A Study of Lexical and Semantic Language Models Applied to Information Retrieval and Learning to Rank
#### Topics:
In a typical IR task we are interested in finding a (usually ranked) list of results that satisfy the information need of a user expressed by means of a query. Many difficulties arise as the satisfaction and information need of the user cannot be directly observed, and thus queries from users can be interpreted in many ways. Moreover, the query is merely a linguistic representation of the actual information need of the user and the gap between them can not be measured either. 

In this lab we study three families of models that are used to measure and rank the relevance of a set of documents given a query: lexical models, semantic models, and machine learning-based re-ranking algorithms that build on top of the former models.


## Copyright

Copyright © 2019 Davide Belli.

<p align=“justify”>
This project is distributed under the <a href="LICENSE">MIT license</a>.  
Please follow the <a href="http://student.uva.nl/en/content/az/plagiarism-and-fraud/plagiarism-and-fraud.html">UvA regulations governing Fraud and Plagiarism</a> in case you are a student.
</p>