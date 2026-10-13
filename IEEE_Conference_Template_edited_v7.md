# Adaptive-IPI: A Teacher-Guided Adaptive Curriculum for Lightweight Indirect Prompt Injection Detection in AI Email Assistants

Author One Author Two Author Three

Department of Computer Science and Engineering

University Name City, Country

{author1, author2, author3}@university.edu

## Faculty Mentor

Dr. Mentor Name Assistant Professor

Department of Computer Science and Engineering

University Name City, Country

mentor@university.edu

Abstract—Large Language Models (LLMs) are increasingly deployed as autonomous agents capable of interacting with ex- ternal tools, databases, and web services. While these capabilities significantly enhance their utility, they also expose LLM-based systems to Indirect Prompt Injection (IPI) attacks, in which malicious instructions embedded within retrieved or third-party content manipulate the model’s behavior without altering the original user query. Existing detection methods typically rely on computationally intensive large language models or lightweight classifiers trained using static datasets, resulting in limited generalization against evolving attack strategies.

This paper presents Adaptive-IPI, a lightweight detection framework that combines adaptive curriculum learning with teacher-guided knowledge distillation to improve the robustness of indirect prompt injection detection while maintaining low inference latency. A Qwen-based teacher model progressively generates increasingly challenging training examples according to the student’s learning progress. The generated curriculum enables the student model, implemented using ModernBERT, to learn discriminative semantic representations from both straight- forward and complex prompt injection patterns. Knowledge dis- tillation further transfers the teacher’s semantic understanding into the lightweight student through a combination of supervised classification and soft-label alignment.

The proposed framework is evaluated on the BIPIA Email QA benchmark, a challenging dataset designed for indirect prompt injection detection in retrieval-augmented language model ap- plications. Experimental results demonstrate an accuracy of 96.54%, precision of 99.57%, recall of 96.81%, F1-score of 98.17%, AUROC of 98.01%, and AUPRC of 99.91%, while maintaining an Expected Calibration Error (ECE) of 0.119. Furthermore, augmenting the benign training distribution with 1,000 curated enterprise emails reduces the false positive rate on unseen enterprise communications from 98.81% to 0.00%, validated across three independent random seeds. The results indicate that adaptive curriculum generation combined with benign distribution expansion substantially improves robustness against diverse prompt injection strategies without sacrificing inference efficiency. Owing to its lightweight architecture and FastAPI-based deployment pipeline, Adaptive- IPI is well suited for practical integration into real-time LLM applications requiring secure and scalable prompt filtering.

Recent benchmark studies have demonstrated that even state-of-the-art language models remain susceptible to indirect prompt injection attacks despite advances in alignment and instruction tuning. Existing mitigation strategies primarily rely

Index Terms—Large Language Models, Indirect Prompt Injec- tion, Prompt Injection Detection, Curriculum Learning, Knowl- edge Distillation, ModernBERT, Qwen, BIPIA, LLM Security, FastAPI

## I. INTRODUCTION

Large Language Models (LLMs) are rapidly being integrated into enterprise productivity tools, enabling AI assistants to answer user questions over emails, documents, and other organizational knowledge sources. Among these applications, **AI Email Question Answering (Email QA)** assistants have emerged as a particularly valuable use case, allowing users to retrieve information from large email repositories using natural language. However, this capability also exposes a critical security vulnerability known as **Indirect Prompt Injection (IPI)**, in which malicious instructions embedded within retrieved emails manipulate the assistant's behavior without altering the user's original query.

Unlike direct prompt injection, where an attacker interacts with the model explicitly, indirect prompt injection exploits the retrieval pipeline itself. An attacker embeds adversarial instructions inside an email that may later be retrieved as context for an unrelated user query. If these instructions are followed by the language model, the assistant may ignore its original task, leak confidential information, or execute unintended actions. Because enterprise email systems routinely process untrusted external communications, robust IPI detection is an essential requirement for safe deployment.

Existing defenses generally fall into two categories. Large-LLM-based defenses provide strong reasoning capability but incur substantial computational cost and are often impractical for latency-sensitive or resource-constrained deployments. Lightweight classifier-based approaches offer efficient inference but are typically trained on static datasets, limiting their ability to adapt to difficult attack patterns and to the linguistic diversity of legitimate enterprise communications.

To address these limitations, we propose **Adaptive-IPI**, a lightweight indirect prompt injection detection framework for AI Email Question Answering assistants. The framework combines teacher-guided knowledge distillation with an adaptive hard-negative curriculum. A frozen teacher model supervises a ModernBERT student during distillation and subsequently analyzes the student's failure cases to generate targeted hard-negative examples for a single adaptive retraining stage. In addition, we investigate **benign distribution expansion** by incorporating curated enterprise Email QA-style communications into the training data to improve robustness against false positives while preserving benchmark performance.

The primary contributions of this work are summarized as follows:

- We propose **Adaptive-IPI**, a lightweight indirect prompt injection detection framework that combines teacher-guided knowledge distillation with an adaptive hard-negative curriculum for AI Email Question Answering assistants.
- We introduce an adaptive teacher-guided refinement strategy in which a frozen teacher model generates targeted hard-negative examples from observed student failures, enabling iterative improvement without increasing inference cost.
- We investigate benign distribution expansion using curated enterprise Email QA communications and demonstrate that increasing benign linguistic diversity substantially reduces false positives on unseen enterprise emails while maintaining strong benchmark performance.
- We perform extensive experimental validation, including controlled ablation studies, multi-seed reproducibility analysis, calibration analysis, and out-of-distribution robustness evaluation.


## II. RELATED WORK

### A. Indirect Prompt Injection Attacks

Indirect Prompt Injection (IPI) has emerged as one of the most significant security challenges facing retrieval-augmented and tool-enabled LLM systems. Unlike direct prompt injection, IPI embeds malicious instructions within external content such as emails, documents, or web pages, causing the model to execute attacker-controlled instructions when that content is retrieved as context. Recent benchmarks, including **BIPIA**, have provided standardized evaluation protocols for measuring robustness against these attacks in realistic application domains such as Email Question Answering.

### B. Lightweight Detection Approaches

Existing defenses range from reasoning-intensive LLM-based systems to lightweight discriminative classifiers. Large-LLM defenses often achieve strong robustness but introduce substantial computational overhead, making them difficult to deploy in latency-sensitive enterprise environments. Lightweight encoder-based detectors provide significantly lower inference cost, but are commonly trained using static supervision, limiting their ability to adapt to difficult attack patterns and evolving prompt injection strategies.

### C. Knowledge Distillation for Efficient Security Models

Knowledge Distillation (KD) enables compact student models to inherit knowledge from larger teacher models while maintaining efficient inference. Prior work has demonstrated KD's effectiveness across natural language understanding and security-related classification tasks. However, most KD pipelines rely on fixed datasets and perform a single offline transfer of knowledge. They do not explicitly exploit student failure cases to improve subsequent supervision.

### D. Position of This Work

Adaptive-IPI differs from prior lightweight detection approaches in two important ways. First, a frozen teacher model is used not only for distillation but also to analyze student prediction failures and generate targeted hard-negative examples for a single adaptive refinement stage. Second, we investigate **benign distribution expansion** by augmenting the training data with diverse enterprise Email QA communications, demonstrating that richer benign supervision substantially improves robustness to unseen enterprise emails while preserving benchmark performance. Rather than proposing a new encoder architecture, this work focuses on improving the training methodology of lightweight detectors through adaptive supervision and controlled expansion of the benign training distribution.


## III. METHODOLOGY

### A. System Overview

Adaptive-IPI is a lightweight indirect prompt injection detection framework designed for AI Email Question Answering (Email QA) assistants. The framework follows a teacher–student paradigm in which a frozen, high-capacity teacher model supervises a compact ModernBERT student. Unlike conventional knowledge distillation pipelines that rely on a fixed training corpus, Adaptive-IPI performs a single adaptive refinement cycle driven by the student's observed failure cases.

The complete workflow consists of six stages:

1. Dataset construction from the BIPIA Email QA benchmark.
2. Teacher annotation using a frozen Qwen model.
3. Initial student training through knowledge distillation.
4. Failure analysis on the validation set.
5. Teacher-guided hard-negative generation targeting observed student errors.
6. Final student retraining on the expanded dataset.

At deployment time, only the lightweight ModernBERT student is used for inference. The teacher model is never required during deployment.

### B. Threat Model

We consider an AI Email Question Answering assistant that retrieves email content to answer a user's query. The attacker cannot modify the user's prompt or the deployed model, but can embed malicious natural-language instructions within emails that may later be retrieved as contextual evidence. The objective of the detector is to classify retrieved email content as either **Benign** or **Indirect Prompt Injection** before it is consumed by the downstream LLM.

### C. Teacher-Guided Knowledge Distillation

A frozen Qwen teacher provides supervision for the student by producing soft probability distributions and binary labels over the BIPIA Email QA training set. The ModernBERT student is optimized using knowledge distillation to learn the teacher's decision boundary while retaining efficient inference suitable for CPU deployment.

### D. Adaptive Hard-Negative Curriculum

After the initial student converges, validation errors are analyzed to identify systematic false positives and false negatives. These failure cases are supplied to the frozen teacher, which generates targeted hard-negative training examples specifically designed to expose weaknesses in the current student model. The generated samples are merged with the original training corpus and the student is retrained for a single adaptive refinement stage. This targeted curriculum focuses additional supervision on difficult decision boundaries instead of uniformly generating synthetic data.

### E. Benign Distribution Expansion

Although the adaptive curriculum improves robustness against prompt injection attacks, lightweight detectors remain sensitive to benign enterprise communications that differ from the original benchmark distribution. To investigate this limitation, we augment the benign training distribution with curated enterprise Email QA-style communications from the Enron corpus. This controlled augmentation increases benign linguistic diversity while preserving the original attack distribution, enabling a systematic study of false-positive robustness.

### F. Inference Pipeline

During inference, an incoming email retrieved by the Email QA system is tokenized and processed by the trained ModernBERT student. The classifier outputs a binary prediction together with an associated confidence score. Emails predicted as prompt injections are blocked before reaching the downstream LLM, while benign emails are forwarded to the answering pipeline. Because the teacher participates only during offline training, inference remains lightweight and suitable for practical deployment.


## IV. EXPERIMENTAL SETUP

### A. Datasets

The primary evaluation is conducted on the **BIPIA Email Question Answering benchmark**, using its official train, validation, and held-out test partitions. The benchmark contains benign enterprise emails together with indirect prompt injection attacks designed for Email QA systems.

To evaluate robustness beyond the benchmark distribution, we perform an additional benign distribution expansion study using curated enterprise Email QA-style communications extracted from the Enron corpus. Controlled augmentation sets containing 250, 500, and 1000 benign emails are constructed for ablation experiments. Evaluation is performed on a disjoint held-out Enron subset after removing duplicate and near-duplicate samples from the training partition.

### B. Models

The teacher model is **Qwen**, used exclusively during offline training for annotation, knowledge distillation, and adaptive hard-negative generation. The deployed detector is **ModernBERT-Base**, selected for its strong language understanding capability and lightweight inference characteristics.

### C. Training Procedure

Training proceeds in two stages. First, the student is trained through teacher-guided knowledge distillation on the BIPIA Email QA benchmark. After convergence, validation failures are analyzed and supplied to the frozen teacher, which generates targeted hard-negative examples. These samples are merged into the training corpus for a single adaptive refinement stage. Benign distribution expansion experiments are subsequently performed by augmenting the benign training distribution with controlled amounts of enterprise Email QA communications.

### D. Evaluation Metrics

Performance is measured using Accuracy, Precision, Recall, F1-score, AUROC, AUPRC, and Expected Calibration Error (ECE). To assess robustness, we additionally report false-positive rates on exploratory handcrafted benign prompts and on the held-out Enron enterprise Email QA evaluation set.

### E. Baselines and Ablation Studies

The proposed framework is evaluated through two primary training configurations, each isolating the contribution of benign distribution expansion:

- **Adaptive-IPI:** The ModernBERT student is trained using teacher-guided knowledge distillation on the BIPIA benchmark, followed by a single adaptive hard-negative refinement stage in which the frozen teacher generates targeted examples from observed student failures. This configuration serves as the primary baseline and demonstrates the core contribution of the adaptive curriculum.
- **Adaptive-IPI + Benign Distribution Expansion:** The full proposed framework, extending the Adaptive-IPI training with augmented benign training data from curated enterprise Email QA communications (Enron250, Enron500, and Enron1000).

Each experiment is evaluated on the same benchmark protocol, and the final configuration (Adaptive-IPI + Enron1000) is further validated through multi-seed reproducibility experiments using independent random initializations.

### F. Implementation Details

All experiments are implemented in Python using PyTorch and Hugging Face Transformers. Training is performed on Apple Silicon using the Metal Performance Shaders (MPS) backend. Unless otherwise stated, identical optimization settings are maintained across comparative experiments to ensure that observed performance differences arise from the proposed methodology rather than changes in training configuration.


## V. RESULTS AND DISCUSSION

This section evaluates Adaptive-IPI through progressively more challenging experimental settings. We first compare the proposed framework against the static KD baseline, followed by reproducibility studies, benign distribution expansion experiments, and out-of-distribution (OOD) robustness analysis.

### A. Comparison of Progressive Teacher-Guided Training Strategies

**[TABLE II: Comparison of Progressive Training Strategies]**

*Insert Table II showing Static KD, Adaptive-IPI (no benign expansion), and Adaptive-IPI + Enron1000, reporting Accuracy, Precision, Recall, F1-score, AUROC, AUPRC, and ECE, along with OOD FPR.*

Table II presents a staged evaluation isolating the contribution of each proposed component on the held-out test set. The initial Static KD baseline (trained on the BIPIA dataset without curriculum augmentation) establishes strong benchmark recall but exhibits vulnerabilities to false positives, both in-domain and out-of-distribution. 

Integrating the adaptive hard-negative curriculum (Adaptive-IPI) shifts the performance trade-offs, improving benchmark precision and AUROC while yielding a slight decrease in overall F1-score (98.48% to 97.07%) and failing to fully resolve false positives on unseen enterprise communications (98.81% Enron FPR). 

Finally, Benign Distribution Expansion (Adaptive-IPI + Enron1000) achieves the strongest overall balance. Augmenting the training distribution with curated enterprise Email QA communications recovers benchmark F1-score (98.17%) while simultaneously eliminating false positives on the held-out enterprise evaluation set (0.00% Enron FPR). This demonstrates that while the adaptive curriculum effectively distills nuanced decision boundaries from the teacher, exposing the student to authentic enterprise distributions is essential for robust, low-false-positive deployment.

---

### B. Multi-Seed Reproducibility

**[TABLE III PLACEHOLDER: Multi-Seed Results]**

*Insert the reproducibility table (Seeds 42–45) reporting mean ± standard deviation for Accuracy, Precision, Recall, F1-score, AUROC, AUPRC, and ECE.*

The low variance observed across independent random initializations indicates that the proposed training pipeline is stable and does not rely on favorable random weight initialization.

---

### C. Benign Distribution Expansion

**[TABLE IV PLACEHOLDER: Enron250 / Enron500 / Enron1000 Ablation]**

*Insert the benign distribution expansion table including BIPIA metrics, Handcrafted FPR, and Held-out Enron FPR.*

To evaluate whether expanding the benign training distribution improves robustness beyond the benchmark, we augmented the BIPIA training data with curated enterprise Email QA communications. Increasing benign linguistic diversity substantially reduced false positives on unseen enterprise emails while maintaining strong benchmark performance.

---

### D. Out-of-Distribution Robustness

**[FIGURE 4 PLACEHOLDER: OOD False Positive Comparison]**

*Insert a bar chart comparing Handcrafted Benign FPR and Held-out Enron FPR across all configurations.*

The proposed framework demonstrates improved robustness to enterprise Email QA communications compared with the baseline. Although challenging handcrafted prompts remain a difficult OOD setting, the results indicate that benign distribution expansion addresses an important class of false positives encountered in enterprise environments.

---

### E. Calibration Analysis

**[FIGURE 5 PLACEHOLDER: Reliability Diagram]**

**[TABLE V PLACEHOLDER: Calibration Metrics]**

Expected Calibration Error (ECE) is reported to assess the reliability of predicted confidence scores. While benign distribution expansion slightly changes calibration characteristics, the detector maintains competitive confidence estimates while substantially improving robustness to benign enterprise communications.

---

### F. Discussion

The experimental results support three primary observations.

1. Teacher-guided knowledge distillation provides stronger supervision than binary labels alone.
2. Adaptive hard-negative refinement improves robustness by targeting observed student failures rather than relying on a static training distribution.
3. Benign distribution expansion significantly reduces false positives on enterprise Email QA communications, demonstrating that lightweight detectors benefit from richer benign supervision.

The remaining false positives observed on exploratory handcrafted prompts highlight opportunities for future work in broader out-of-distribution generalization. However, the proposed framework consistently maintains strong benchmark performance while improving practical robustness for enterprise Email QA deployment.


## VI. CONCLUSION AND FUTURE WORK

This paper presented **Adaptive-IPI**, a lightweight indirect prompt injection detection framework for AI Email Question Answering assistants. By combining teacher-guided knowledge distillation with an adaptive hard-negative curriculum, the proposed framework improves the robustness of a compact ModernBERT student while preserving lightweight inference suitable for practical deployment. Unlike conventional static knowledge distillation pipelines, Adaptive-IPI performs a targeted adaptive refinement stage in which a frozen teacher analyzes student failure cases and generates hard-negative examples to strengthen the student's decision boundary.

Beyond adaptive curriculum learning, this work investigated **benign distribution expansion** as a complementary strategy for reducing false positives. Controlled augmentation with enterprise Email QA communications demonstrated that increasing benign linguistic diversity can substantially improve robustness to unseen enterprise emails while maintaining strong benchmark performance.

Extensive experiments, including baseline comparisons, controlled ablation studies, multi-seed reproducibility analysis, calibration evaluation, and out-of-distribution robustness experiments, demonstrate the effectiveness of the proposed framework. Together, these results suggest that adaptive supervision and richer benign training distributions are complementary mechanisms for improving lightweight prompt injection detectors.

### Future Work

Several promising research directions remain:

- Evaluate the framework on additional Email QA benchmarks and enterprise communication corpora.
- Investigate confidence calibration techniques such as post-hoc temperature scaling.
- Explore multiple adaptive refinement iterations while preserving lightweight deployment.
- Study integration with complementary defenses such as intent reconstruction, retrieval verification, or downstream policy enforcement.
- Extend the methodology to other retrieval-based LLM applications beyond Email Question Answering.

Adaptive-IPI demonstrates that carefully designed training methodology, rather than increasing model size alone, can substantially improve the practical robustness of lightweight indirect prompt injection detection systems.
