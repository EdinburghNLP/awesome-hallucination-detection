# awesome-hallucination-detection

## Papers and Summaries

### [Elastic Weight Removal for Faithful and Abstractive Dialogue Generation](https://arxiv.org/abs/2303.17574)
- **Metrics:** Faithfulness between predicted response and ground-truth knowledge (Tab. 1) -- Critic, Q², BERT F1, F1.
- **Datasets:** Wizard-of-Wikipedia (WoW), the DSTC9 and DSTC11 extensions of MultiWoZ 2.1, FaithDial -- a de-hallucinated subset of WoW.

### [Trusting Your Evidence: Hallucinate Less with Context-aware Decoding](https://arxiv.org/abs/2305.14739)
- **Metrics:** Factual consistency of summaries: BERT-Precision and FactKB. MemoTrap and NQ-Swap: Exact Match.
- **Datasets:** Summarisation: CNN-DM, XSUM. Knowledge Conflicts: MemoTrap, NQ-Swap.

### [When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories](https://arxiv.org/2212.10511)
- **Metrics:** Exact Match/Accuracy.
- **Datasets:** QA datasets with long-tail entities: PopQA, EntityQuestions; NQ.

### [Retrieval Augmentation Reduces Hallucination in Conversation](https://arxiv.org/2104.10511)
- **Metrics:** Generation: Perplexity, Unigram Overlap (F1), BLEU-4, ROUGE-L. Overlap between generation and knowledge on which the human grounded during dataset collection: Knowledge F1; only consider words that are infrequent in the dataset when calculating F1: Rare F1.
- **Datasets:** Wow, CMU Document Grounded Conversations (CMU_DoG). Knowledge source: KiLT Wikipedia dump.

### 2. [Just Ask for Calibration](https://arxiv.org/abs/2305.14975)
- **Metrics:** Evaluation of confidence scores from LLMs; average accuracy based on confidence score ranges; calibration measured using multiple metrics including ECE.
- **Datasets:** SciQ science question-answering dataset; TruthfulQA for testing common misconceptions; TriviaQA for reading comprehension.

### 3. [[Possible Title: ACL 2023]](https://arxiv.org/abs/2212.10511)
- **Metrics:** Wikipedia page views for popularity; QA accuracy on open-domain benchmarks.
- **Datasets:** New dataset POPQA with 14k questions on long-tail entities; EntityQuestions and Natural Questions datasets.

### 4. [Trusting Your Evidence:](https://arxiv.org/abs/2305.14739)
- **Metrics:** BERT-Precision for factual summary consistency; Exact Match for QA tasks.
- **Datasets:** LLaMA-30B for knowledge conflicts QA; NQ-Swap based on Natural Questions.

### 5. [How Language Model Hallucinations Can Snowball](https://arxiv.org/abs/2305.13534)
- **Metrics:** Percentage of Wrong Answers (Hallucinations) and cases where "the model knows it's wrong" (Snowballed Hallucinations).
- **Datasets:** Primality Testing, Senator Search, Graph Connectivity.

### 6. [Improving Language Models with](https://arxiv.org/abs/2305.14718)
- **Metrics:** Use of reinforcement learning to measure LM's desired behavior without intensive data manipulation; BERT-Precision for factual summary consistency.
- **Datasets:** Chatbot corpus of comment pairs; Wizard of Wikipedia (WoW) dataset and its FaithDial version.

### 7. [Generating with Confidence: Uncertainty](https://arxiv.org/abs/2305.19187)
- **Metrics:** Metric for average semantic dispersion; Importance of reliable uncertainty measure; Need for informative confidence/uncertainty metric.
- **Datasets:** CoQA for open-book conversational QA; TriviaQA and Natural Questions (NQ) for closed-book QA.

