# awesome-hallucination-detection

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/EdinburghNLP/awesome-hallucination-detection) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Papers and Summaries

### [Attention Satisfies: A Constraint-Satisfaction Lens on Factual Errors of Language Models](https://arxiv.org/abs/2309.15098)
- **Metrics:** AUROC, risk-coverage curve operating points
- **Datasets:** CounterFact, factual queries generated from Wikidata
- **Comments:** This paper models factual queries as constraint-satisfaction problems and find that attention to constraint tokens significantly correlates with factual correctness / hallucinations.

### [TRUE: Re-Evaluating Factual Consistency Evaluation](https://arxiv.org/abs/2204.04991)
- **Metrics:** AUROC, across multiple datasets and evaluation methods
- **Datasets:** PAWS, XSum, QAGS, FRANK, SummEval, BEGIN, Q^2, DialFact, FEVER, VitaminC

### [TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models](https://arxiv.org/abs/2305.11171)
- **Metrics:** AUROC, across multiple datasets and evaluation methods
- **Datasets:** XSum, QAGS, FRANK, SummEval

### [SAC$`^3`$: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency](https://arxiv.org/abs/2311.01740)
- **Metrics:** Accuracy and AUROC: classification QA and open-domain QA
- **Datasets:** Prime number and senator search from Snowball Hallucination, HotpotQA and Nq-open QA

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
- **Datasets:** Faithful Knowledge-Grounded Dialog: FaithDial, a more faithful subset of WoW.

### [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/abs/2305.19187)
- **Metrics:** AUROC, AUARC, Uncertainty and Confidence metrics (NumSet, Deg, EigV).
- **Datasets:** CoQA (Open-book Conversational QA dataset), TriviaQA and Natural Questions (Closed-book QA).

### [FaithDial: A Faithful Benchmark for Information-Seeking Dialogue](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00529/114373/FaithDial-A-Faithful-Benchmark-for-Information)
- **Metrics:** Metrics measure either the degree of hallucination of generated responses wrt to some given knowledge or their overlap with gold faithful responses: Critic, Q² (F1, NLI), BERTScore, F1, BLEU, ROUGE.
- **Datasets:** FaithDial, WoW.

### [Neural Path Hunter: Reducing Hallucination in Dialogue Systems via Path Grounding](https://arxiv.org/abs/2104.08455)
- **Metrics:** FeQA, a faithfulness metric; Critic, a hallucination critic; BLEU.
- **Datasets:** OpenDialKG, a dataset that provides open-ended dialogue responses grounded on paths from a KG.

### [HaluEval: A Large-Scale Hallucination Evaluation Benchmark](https://arxiv.org/abs/2305.11747)
- **Metrics:** Accuracy: QA, Dialogue, Summarisation.
- **Datasets:** HaluEval, a collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognising hallucinations.

### [Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation](https://arxiv.org/abs/2305.15852)
- **Metrics:** After generating sentence pairs, it measures precision, recall, and F1 score in detection tasks.
- **Datasets:** 12 selected topics from Wikipedia.

### [Mitigating Language Model Hallucination with Interactive Question-Knowledge Alignment](https://arxiv.org/abs/2305.13669)
- **Metrics:** *Coverage*: a binary metric that determines whether all the correct gold answer values are included in the generated value. *Hallucination*: a binary indicator that assesses the presence of generated values that do not exist in the question values and gold grounding values. *User Simulator*: user simulator as an "oracle" language model with access to attribution information about the target answer.
- **Datasets:** FuzzyQA, a dataset based on HybridDialogue and MuSiQue where complex questions were simplified using ChatGPT.

### [Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback](https://arxiv.org/abs/2302.12813)
- **Metrics:** KF1, BLEU, ROUGE, chrF, METEOR, BERTScore, BARTScore, BLEURT, Avg length.
- **Datasets:** News Chat: DSTC7 Track 2 was repurposed as an evaluation corpus for news conversation. Customer Service: uses DSTC11 Track 5 as a showcase in a conversational customer service scenario, expanding upon DSTC9 Track 1 by incorporating subjective information.

### [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896)
- **Metrics:** Sentence-level Hallucination Detection (AUC-PR), and Passage-level Hallucination Detection (Pearson and Spearman's correlation coefficients).
- **Datasets:** Generated Wikipedia articles from WikiBio, with annotated hallucinations.

### [The Internal State of an LLM Knows When it's Lying](https://arxiv.org/abs/2304.13734)
- **Metrics:** Per-topic and average accuracy.
- **Datasets:** The True-False Dataset contains true and false statements covering several topics -- Cities, Inventions, Chemical Elements, Animals, Companies, and Scientific Facts.

### [Chain of Knowledge: A Framework for Grounding Large Language Models with Structured Knowledge Bases](https://arxiv.org/abs/2305.13269)
- **Metrics:** Exact Match.
- **Datasets:** FEVER, Adversarial HotpotQA.

### [Halo: Estimation and Reduction of Hallucinations in Open-Source Weak Large Language Models](https://arxiv.org/abs/2308.11764)
- **Metrics:** HaloCheck and SelfCheckGPT scores; consistency, factuality.
- **Datasets:** Generated and reviewed questions in the NBA domain.

### [A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation](https://arxiv.org/abs/2307.03987)
- **Metrics:** Precision and Recall when detecting Sentence-level and Concept-level Hallucinations.
- **Datasets:** ChatGPT-generated paragraphs spanning 150 topics from diverse domains.

### [Sources of Hallucination by Large Language Models on Inference Tasks](https://arxiv.org/abs/2305.14552)
- **Metrics:** Directional Levy/Holt precision and recall with entity insertions and replacements.
- **Datasets:** Levy/Holt dataset, containing premise-hypothesis pairs with a task formatted as *Given [premise P], is it true that [hypothesis H]?*, where the model is evaluated with random premises.

### [Hallucinations in Large Multilingual Translation Models](https://arxiv.org/abs/2303.16104)
- **Metrics:** Rate to which MT system produces hallucinations under perturbation (Language Pair fraction, rate).
- **Datasets:** Flores-101, WMT, TICO.

### [Citation: A Key to Building Responsible and Accountable Large Language Models](https://arxiv.org/abs/2307.02185)
- **Metrics:** N/A
- **Datasets:** N/A

### [Zero-Resource Hallucination Prevention for Large Language Models](https://arxiv.org/abs/2309.02654)
- **Metrics:** Hallucinatory instruction classification: AUC, ACC, F1, PEA.
- **Datasets:** Concept-7, which focuses on classifying potential hallucinatory instructions.

### [RARR: Researching and Revising What Language Models Say, Using Language Models](https://arxiv.org/abs/2210.08726)
- **Metrics:** Attributable to Identified Sources (AIS) scores before and after editing.
- **Datasets:** Generated statements by creating task inputs from three datasets and prompting different models to produce long-form outputs which may contain hallucinations -- Factoid statements, Reasoning chains, and Knowledge-intensive dialogues.

### [Q²: Evaluating Factual Consistency in Knowledge-Grounded Dialogues via Question Generation and Question Answering](https://arxiv.org/abs/2104.08202)
- **Metrics:** Q² is a metric itself, and it is compared with F1 token-level overlap, Precision and Recall, Q² w/o NLI, E2E NLI, Overlap, BERTScore, and BLEU.
- **Datasets:** WoW which contains dialogues in which a bot needs to respond to useri nputs in a knowledgeable way; Topical-Chat, a human-human knowledge-grounded conversation dataset; Dialogue NLI, a dataset based on the Persona-Chat dialogue task consisting of premise-hypothesis pairs.

### [Do We Know What We Don’t Know? Studying Unanswerable Questions beyond SQuAD 2.0](https://aclanthology.org/2021.findings-emnlp.385.pdf)
- **Metrics:** EM on All, "Has answer", and "IDK"
- **Datasets:** MNLI, SQuAD 2.0, ACE-whQA.

### [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495)
- **Metrics:** Wikidata and Wiki-Category List: test precision, average number of positive and negative (hallucination) entities for list-based questions; MultiSpanQA: F1, Precision, Recall; Longform generation of biographies: FactScore.
- **Datasets:** Wikidata, Wiki-Category List, MultiSpanQA, Longform Generation of Biographies.

### [Detecting and Mitigating Hallucinations in Multilingual Summarisation](https://arxiv.org/abs/2305.13632)
- **Metrics:** mFACT, a novel multilingual faithful metric developed from four English faithfulness metrics: DAE, QAFactEval, ENFS%, and EntFA.
- **Datasets:** XL-Sum, a multilingual summarisation dataset.

### [Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization](https://aclanthology.org/2022.acl-long.236/)
- **Metrics:** XEnt: Hallucination (Accuracy, F1), Factuality (Accuracy, F1), ROUGE, % of novel n-gram, Faithfulness (%ENFS, FEQA, DAE), EntFA (% Factual Ent., % Factual Hal.)
- **Datasets:** A novel dataset, XEnt, for analysing entity hallucination and factuality in abstractive summarisation, consisting of 800 summaries generated by BART and annotated. MEnt, a set of factuality and hallucination annotations for XSum.
- **Comments:** Tab. 2 outlines several types of hallucinations (e.g., factual, non-factual, intrinsic).

### [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627)
- **Metrics:** Fluency (MAUVE), Correctness (EM recall for ASQA, recall-5 for QAMPARI, claim recall for ELI5), Citation quality (citation recall, citation precision).
- **Datasets:** QA datasets such that 1) they contain factual questions in which references are important, 2) questions require long-text answers covering multiple aspects, and 3) answering the questions requires synthesising multiple sources: ASQA, QAMPARI, ELI5.

### [A Token-level Reference-free Hallucination Detection Benchmark for Free-form Text Generation](https://arxiv.org/abs/2104.08704)
- **Metrics:** Acc, G-Mean, BSS, AUC, Not Hallucination (P, R, F1), Hallucination (P, R, F1).
- **Datasets:** HaDes (HAllucination DEtection dataSet), a novel token-level reference-free annotated hallucination detection dataset obtained by perturbing a large number of text segments extracted from the English Wikipedia and verified with crowd-sourced annotations.
- **Comments:** Fig. 3 outlines several hallucination types (domain-specific knowledge, commonsense knowledge, incoherence or improper collocation, unrelated to central topic, conflict with preceding context, conflict with succeeding context, ..)

### [Generating Benchmarks for Factuality Evaluation of Language Models](https://arxiv.org/abs/2307.06908)
- **Metrics:** Percentage of examples it assigns the highest probability to the factual completion.
- **Datasets:** Wiki-FACTOR and News-FACTOR: two novel factuality evaluation benchmarks for LLMs, based on Wikipedia and News articles. Each example consists of a prefix, a factual completion and three similar but non-factual alternatives.
- **Comments:** The paper introduces a framework for automatically generating such datasets from a given corpus, detailed in Section 3.

### [Do Language Models Know When They're Hallucinating References?](https://arxiv.org/abs/2305.18248)
- **Metrics:** Hallucination rate (H%, out of 1000 generated titles)
- **Datasets:** Generated (true and hallucinated) references on topics from the ACM Computing Classification System.

### [Why Does ChatGPT Fall Short in Providing Truthful Answers?](https://arxiv.org/abs/2304.10513)
- **Metrics:** #Correct and #Wrong answers, and different type of failure counts: Comprehension, Factualness, Specificity, Inference.
- **Datasets:** HotpotQA, BoolQ
- **Comments:** This has a nice taxonomy on different error types -- e.g., *comprehension*, *factualness*, *specifity*, *inference*.

### [LM vs LM: Detecting Factual Errors via Cross Examination](https://arxiv.org/abs/2305.13281)
- **Metrics:** Precision, Recall, F1 (under different cross-examination strategies: AYS, IDK, Confidence-Based, IC-IDK)
- **Datasets:** TriviaQA, NQ, PopQA

### [RHO (ρ): Reducing Hallucination in Open-domain Dialogues with Knowledge Grounding](https://arxiv.org/abs/2212.01588)
- **Metrics:** BLEU, ROUGE-L; FeQA, QuestEval, EntityCoverage (Precision, Recall, F1) to estimate the hallucination degree -- FrQA and QuestEval are QA-based metrics for evaluating the faithfulness of the output in the generation task.
- **Datasets:** OpenDialKG

### [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251)
- **Metrics:** %Supported statements across varying frequency levels of human entities.
- **Datasets:** People biographies generated from LLMs, where human annotators break them into supporting facts.

### [ExpertQA: Expert-Curated Questions and Attributed Answers](https://arxiv.org/abs/2309.07852)
- **Metrics:** zero-shot (P, R, F1) and fine-tuned (P, R, F1) of AutoAIS labels; FActScore F1 scores on reference factuality labels; AutoAIS (Attributable to Identified Sources) scores.
- **Datasets:** Expert-curated questions across multiple fields (e.g., Anthropology, Architecture, Biology, Chemistry, Engineering & Technology, Healthcare/Medicine; see Tab. 1 for a sample) organised by Question Type (e.g., Directed question with single unambiguous answer, open-ended potentially ambiguous question, summarisation of information of a topic, advice or suggestion on how to approach a problem; see Tab. 2)

### [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883)
- **Metrics:** TruthffulQA: MC1, MC2, MC3 scores; FACTOR: News, Wiki; these were multiple-choice results. Open-ended generation: for TruthfulQA, they use %Truth, %Info, %Truth*Info, %Reject; for CoT tasks (StrategyQA and GSM8K) they go with accuracy.
- **Datasets:** TruthfulQA, FACTOR (news/wiki), StrategyQA, GSM8K

### [FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation](https://arxiv.org/abs/2310.03214)
- **Metrics:** Accuracy (Strict, Relaxed on Fast-changing questions, Slow-changing questions, Never-changing questions, False-premise questions involve knowledge before 2022 and since 2022, 1-hop and multi-hop questions, and Overall).
- **Datasets:** FreshQA, a new QA benchmark with 600 questions covering a wide spectrum of question and answer types.


### [Med-HALT: Medical Domain Hallucination Test for Large Language Models](https://arxiv.org/abs/2307.15343)
- **Metrics:** Reasoning Hallucination Tests (False Confidence Tests, None of the Above Tests, Fake Questions Tests), Memory Hallucination Tests (Abstract-to-Link Tests, PMID-to-Title Tests, Title-to-Link Tests, Link-to-Title Tests); Accuracy, Pointwise Score.
- **Datasets:** Med-HALT: MEDMCQA, Headqa, Medqa USMILE, Medqa (Taiwan), Pubmed.

### [Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators](https://arxiv.org/abs/2310.07289)
- **Metrics:** Factuality, Relevance, Coherence, Informativeness, Helpfulness and Validity.
- **Datasets:** Natural Questions, Wizard of Wikipedia.


## Overviews, Surveys, and Shared Tasks

- [Mitigating LLM Hallucinations: a multifaceted approach](https://amatriain.net/blog/hallucinations)
- [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629)
- [A Survey of Hallucination in Large Foundation Models](https://arxiv.org/abs/2309.05922)
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [SemEval-2024 Task-6 - SHROOM, a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes](https://helsinki-nlp.github.io/shroom/)
- [llm-hallucination-survey](https://github.com/HillZhang1999/llm-hallucination-survey)
- [How Do Large Language Models Capture the Ever-changing World Knowledge? A Review of Recent Advances](https://arxiv.org/abs/2310.07343)

## Taxonomies

[Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) classifies metrics in *Statistical* (ROUGE, BLEU, PARENT, Knowledge F1, ..) and *Model-based* metrics. The latter are further structured in the following classes:
- **Information-Extraction (IE)-based**: retrieve an answer from a knowledge source and compare it with the generated answer -- there might be problems due to the error propagation from the IE model.
- **QA-based**: measure the overlap/consistency between generation and source reference, based on the intuition that similar answers will be generated from the same question if the generation is factually consistent with the source reference. Used to evaluate hallucinations in summarisation, dialogue, and data2text generation. Composed of a *question generation* model and a *question answering* model.
- **Natural Language Inference (NLI)-based**: based on the idea that only the source knowledge reference should entail the entirety of the information in faithful and hallucination-free generation.

[A Survey of Hallucination in “Large” Foundation Models](https://arxiv.org/abs/2309.05922) surveys papers flagging them for *detection*, *mitigation*, *tasks*, *datasets*, and *evaluation metrics*. Regarding hallucinations in text, it categorises papers by *LLMs*, *Multilingual LLMs*, and *Domain-specific LLMs*.

## Measuring Hallucinations in LLMs
- [AnyScale - Llama 2 is about as factually accurate as GPT-4 for summaries and is 30X cheaper](https://www.anyscale.com/blog/llama-2-is-about-as-factually-accurate-as-gpt-4-for-summaries-and-is-30x-cheaper)
- [Arthur.ai - Hallucination Experiment](https://www.arthur.ai/gap-articles/hallucination-experiment)
- [Vectara - Cut the Bull…. Detecting Hallucinations in Large Language Models](https://vectara.com/cut-the-bull-detecting-hallucinations-in-large-language-models/)
- [Vectara LLM Leaderboard](https://github.com/vectara/hallucination-leaderboard)

## Open Source Models for Measuring Hallucinations
- [AlignScore Code and Model - GutHub](https://github.com/yuh-zha/AlignScore)
- [Google True Teacher Model - HuggingFace](https://huggingface.co/google/t5_11b_trueteacher_and_anli)
- [Hallucination Evaluation Model - HuggingFace](https://huggingface.co/vectara/hallucination_evaluation_model)
- [Summac Code and Model - GitHub](https://github.com/tingofurro/summac)

## Definitions and Notes

### Extrinsic and Intrinsic Hallucinations

[Neural Path Hunter](https://arxiv.org/abs/2104.08455) defines as *extrinsic hallucination* as an utterance that brings a new span of text that does not correspond
to a valid triple in a KG, and as *intrinsic hallucination* as an utterance that misuses either the subject or object in a KG triple such that there is no direct path between the two entities. [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) defines as *extrinsic hallucination* a case where  the generated output that cannot be verified from the source content, and as an *intrinsic hallucination* a case where the generated output contradicts the source content.
