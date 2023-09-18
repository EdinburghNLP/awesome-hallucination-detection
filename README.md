# awesome-hallucination-detection

## Papers and Summaries

### [Elastic Weight Removal for Faithful and Abstractive Dialogue Generation](https://arxiv.org/abs/2303.17574)
- **Metrics:** Faithfulness between predicted response and ground-truth knowledge (Tab. 1) -- Critic, Q², BERT F1, F1.
- **Datasets:** Wizard-of-Wikipedia (WoW), the DSTC9 and DSTC11 extensions of MultiWoZ 2.1, FaithDial -- a de-hallucinated subset of WoW.

### [Trusting Your Evidence: Hallucinate Less with Context-aware Decoding](https://arxiv.org/abs/2305.14739)
- **Metrics:** Factual consistency of summaries: BERT-Precision and FactKB. MemoTrap and NQ-Swap: Exact Match.
- **Datasets:** Summarisation: CNN-DM, XSUM. Knowledge Conflicts: MemoTrap, NQ-Swap.

### [When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories](https://arxiv.org/abs/2212.10511)
- **Metrics:** Exact Match/Accuracy.
- **Datasets:** QA datasets with long-tail entities: PopQA, EntityQuestions; NQ.

### [Retrieval Augmentation Reduces Hallucination in Conversation](https://arxiv.org/abs/2104.07567)
- **Metrics:** Generation: Perplexity, Unigram Overlap (F1), BLEU-4, ROUGE-L. Overlap between generation and knowledge on which the human grounded during dataset collection: Knowledge F1; only consider words that are infrequent in the dataset when calculating F1: Rare F1.
- **Datasets:** Wow, CMU Document Grounded Conversations (CMU_DoG). Knowledge source: KiLT Wikipedia dump.

### [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://arxiv.org/abs/2305.14975)
- **Metrics:** Expected Calibration Error (ECE) with temperature scaling (ECE-t); accuracy@coverage and coverage@accuracy.
- **Datasets:** Question Answering datasets assessing factual knowledge: TriviaQA, SciQ, TruthfulQA.

### [How Language Model Hallucinations Can Snowball](https://arxiv.org/abs/2305.13534)
- **Metrics:** Percentage of Wrong Answers (Hallucinations) and cases where "the model knows it's wrong" (Snowballed Hallucinations).
- **Datasets:** Primality Testing, Senator Search, Graph Connectivity.

### [Improving Language Models with Advantage-based Offline Policy Gradients](https://arxiv.org/abs/2305.14718)
- **Metrics:** Faithfulness evaluation for Knowledge-Grounded response generation on FaithDial -- FaithCritic, CoLA (Fluency), Dialog Engagement, Length-penalised TF-IDF Diversity. 
- **Datasets:** Faithful Knowledge-Grounded Dialog: FaithDial, a more faithgul subset of WoW.

### [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/abs/2305.19187)
- **Metrics:** AUROC, AUARC, Uncertainty and Confidence metrics (NumSet, Deg, EigV).
- **Datasets:** CoQA (Open-book Conversational QA dataset), TriviaQA and Natural Questions (Closed-book QA).

### [FaithDial: A Faithful Benchmark for Information-Seeking Dialogue](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00529/114373/FaithDial-A-Faithful-Benchmark-for-Information)
- **Metrics:** Metrics measure either the degree of hallucination of generated responses wrt to some given knowledge, or their overlap with gold faithful responses: Critic, Q² (F1, NLI), BERTScore, F1, BLEU, ROUGE.
- **Datasets:** FaithDial, WoW.

### [Neural Path Hunter: Reducing Hallucination in Dialogue Systems via Path Grounding](https://arxiv.org/abs/2104.08455)
- **Metrics:** FeQA, a faithfulness metric; Critic, a hallucination critic; BLEU.
- **Datasets:** OpenDialKG, a dataset that provides open-ended dialogue responses grounded on paths from a KG.

### [HaluEval: A Large-Scale Hallucination Evaluation Benchmark](https://arxiv.org/abs/2305.11747)
- **Metrics:** Accuracy: QA, Dialogue, Summarisation.
- **Datasets:** HaluEval, a collection of generated and human-annotated hallucinated samples for evaluating the performane of LLMs in recognising hallucinations.

## Taxonomies

[Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) classifies metrics in *Statistical* (ROUGE, BLEU, PARENT, Knowledge F1, ..) and *Model-based* metrics. The latter are further structured in the following classes:
- **Information-Extraction (IE)-based**: retrieve answer from a knowledge source and compare it with the generated answer -- there might be problems due to the error propagation from the IE model.
- **QA-based**: measure the overlap/consistency between generation and source reference, based on the intuition that similar answers will be generated from the same question if the generation is factually consistent with the source reference. Used to evaluate hallucinations in summarisation, dialogue, and data2text generation. Composed by a *question generation* model and a *question answering* model.
- **Natural Language Inference (NLI)-based**: based on the idea that only thr source knowledge reference should entail the entirety of the information in faithful and hallucination-free generation.

## Definitions and Notes

### Extrinsic and Intrinsic Hallucinations

[Neural Path Hunter](https://arxiv.org/abs/2104.08455) defines as *extrinsic hallucination* as an utterance that brings a new span of text that does not correspond
to a valid triple in a KG, and as *intrinsic hallucination* as an utterance that misuses either the subject or object in a KG triple such that there is no direct path between the two entities. [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) defines as *extrinsic hallucination* a case where  the generated output that cannot be verified from the source content, and as an *intrinsic hallucination* a case where the generated output contradicts the source content.
