# awesome-hallucination-detection

## Papers and Summaries

### 1. [Elastic Weight Removal for Faithful and Abstractive Dialogue Generation](https://arxiv.org/abs/2303.17574)
- **Metrics:** Evaluation shows EWR increases faithfulness; Fisher Information Matrix (FIM) used for parameter importance.
- **Datasets:** Synthetic dataset for anti-expert training; WoW dataset subset annotated with the BEGIN framework.

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
- **Metrics:** Evaluation of model accuracy, especially under faulty context conditions.
- **Datasets:** Primality testing dataset; examples include Senator Search and others.

### 6. [Improving Language Models with](https://arxiv.org/abs/2305.14718)
- **Metrics:** Use of reinforcement learning to measure LM's desired behavior without intensive data manipulation; BERT-Precision for factual summary consistency.
- **Datasets:** Chatbot corpus of comment pairs; Wizard of Wikipedia (WoW) dataset and its FaithDial version.

### 7. [Generating with Confidence: Uncertainty](https://arxiv.org/abs/2305.19187)
- **Metrics:** Metric for average semantic dispersion; Importance of reliable uncertainty measure; Need for informative confidence/uncertainty metric.
- **Datasets:** CoQA for open-book conversational QA; TriviaQA and Natural Questions (NQ) for closed-book QA.

