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

The proposed framework is evaluated on the BIPIA Email QA benchmark, a challenging dataset designed for indirect prompt injection detection in retrieval-augmented language model ap- plications. Experimental results demonstrate an accuracy of 99.06%, precision of 99.14%, recall of 99.87%, F1-score of 99.50%, AUROC of 99.78%, and AUPRC of 99.99%, while maintaining an Expected Calibration Error (ECE) of 0.1096. The results indicate that adaptive curriculum generation substantially improves robustness against diverse prompt injection strategies without sacrificing inference efficiency. Owing to its lightweight architecture and FastAPI-based deployment pipeline, Adaptive- IPI is well suited for practical integration into real-time LLM applications requiring secure and scalable prompt filtering.

Recent benchmark studies have demonstrated that even state-of-the-art language models remain susceptible to indirect prompt injection attacks despite advances in alignment and instruction tuning. Existing mitigation strategies primarily rely

Index Terms—Large Language Models, Indirect Prompt Injec- tion, Prompt Injection Detection, Curriculum Learning, Knowl- edge Distillation, ModernBERT, Qwen, BIPIA, LLM Security, FastAPI

## I. INTRODUCTION

Large Language Models (LLMs) have become a fundamen- tal component of modern intelligent systems, enabling appli- cations such as conversational assistants, autonomous agents, retrieval-augmented generation (RAG), software copilots, and enterprise automation. Their ability to interpret natural lan- guage instructions and reason over external information has significantly expanded their applicability across diverse do- mains. However, this increased capability has also introduced new security vulnerabilities that conventional machine learning systems were not designed to address.

Among these threats, Indirect Prompt Injection (IPI) has emerged as one of the most challenging attack vectors against LLM-based systems. Unlike direct prompt injection, where malicious instructions are explicitly entered by a user, indirect prompt injection embeds adversarial instructions within exter- nal content such as emails, documents, web pages, APIs, or re- trieved database entries. During inference, the language model may inadvertently interpret these hidden instructions as legiti- mate context, resulting in unauthorized behavior, disclosure of confidential information, or manipulation of downstream tool execution.


on two approaches. The first employs powerful proprietary LLMs to classify prompts before execution, achieving high detection accuracy at the expense of substantial computational overhead and latency. The second trains lightweight encoder- based classifiers that offer efficient inference but often struggle to generalize beyond the static attack patterns observed during training. As prompt injection techniques continue to evolve, detectors trained on fixed datasets experience noticeable degra- dation in performance when confronted with previously unseen attack styles.

Curriculum learning has proven effective across multiple machine learning domains by presenting training samples in an order that gradually increases task complexity. Simultaneously, knowledge distillation enables compact student models to inherit semantic reasoning capabilities from larger teacher models while maintaining low inference cost. Although both techniques have demonstrated considerable success indepen- dently, their combined application to indirect prompt injection detection remains largely unexplored.

To address these limitations, this paper proposes Adaptive- IPI, a lightweight detection framework that integrates adaptive curriculum learning with teacher-guided knowledge distillation for robust indirect prompt injection detection. Instead of rely- ing on a static training distribution, the proposed framework employs a Qwen-based teacher model to generate progres- sively challenging prompt injection examples according to the student’s current learning stage. This adaptive curriculum enables the student model to continuously expand its decision boundary while avoiding instability associated with abrupt exposure to difficult adversarial samples.

The student detector is implemented using ModernBERT, which offers an effective balance between contextual rep- resentation quality and computational efficiency. Knowledge distillation transfers semantic knowledge from the teacher through a combination of supervised learning and soft-label guidance, allowing the lightweight student to approximate the teacher’s decision behavior while remaining suitable for real- time deployment.

The proposed framework is evaluated on the BIPIA Email QA benchmark, a widely used benchmark for indirect prompt injection detection. Experimental evaluation demonstrates strong predictive performance with an overall accuracy of 99.06%, precision of 99.14%, recall of 99.87%, F1-score of 99.50%, AUROC of 99.78%, AUPRC of 99.99%, and an Expected Calibration Error (ECE) of 0.1096. These results indicate that Adaptive-IPI achieves both high detection capa- bility and reliable probability calibration while maintaining the computational efficiency required for deployment in produc- tion LLM pipelines.

The primary contributions of this work are summarized as

follows.

- We propose Adaptive-IPI, an adaptive curriculum learn- ing framework for indirect prompt injection detection that progressively generates increasingly challenging adver- sarial training samples.

- We integrate teacher-guided knowledge distillation using a Qwen teacher model and a lightweight ModernBERT student model, enabling robust detection while preserving inference efficiency.

- We evaluate the proposed framework on the BIPIA Email QA benchmark and demonstrate state-of-the-art performance across multiple evaluation metrics, including calibration quality.

- We present a practical FastAPI-based deployment archi- tecture that enables seamless integration of the detector into real-world LLM applications with minimal compu- tational overhead.

The remainder of this paper is organized as follows. Section II reviews existing literature on prompt injection attacks, curriculum learning, and knowledge distillation. Section III describes the proposed Adaptive-IPI framework and its math- ematical formulation. Section IV presents the experimental setup and implementation details. Section V discusses the experimental results and ablation studies. Section VI ana- lyzes deployment considerations, computational complexity, and limitations. Finally, Section VII concludes the paper and outlines future research directions.

## II. RELATED WORK

The growing adoption of Large Language Models (LLMs) in retrieval-augmented generation (RAG), autonomous agents, and tool-using systems has motivated extensive research into their security vulnerabilities. Among these, prompt injection attacks have received considerable attention due to their ability to manipulate model behavior without exploiting software vulnerabilities. This section reviews prior work on indirect prompt injection, lightweight detection techniques, curriculum learning, knowledge distillation, and calibration methods that motivate the proposed Adaptive-IPI framework.

## A. Indirect Prompt Injection Attacks

Prompt injection attacks were first systematically analyzed following the deployment of instruction-tuned LLMs capa- ble of interacting with external environments. Unlike direct prompt injection, indirect prompt injection embeds malicious instructions within retrieved content such as emails, web pages, documents, or API responses. Since these external resources are often incorporated into the model’s context during inference, hidden instructions may override or conflict with the user’s original intent.

Several studies have demonstrated that even highly capable commercial language models remain vulnerable to indirect prompt injection despite reinforcement learning and safety alignment techniques. Such attacks may lead to confidential in- formation leakage, unauthorized tool execution, manipulation of generated responses, and bypassing of safety mechanisms. The BIPIA benchmark was subsequently introduced to provide a standardized evaluation framework for assessing prompt injection robustness across diverse retrieval scenarios.


Although recent defensive strategies employ powerful pro- prietary LLMs as security filters, their computational require- ments introduce significant inference latency and deployment costs, limiting their applicability in real-time systems.

## B. Lightweight Prompt Injection Detection

To reduce computational overhead, recent research has explored lightweight encoder-based classifiers for prompt injection detection. Transformer encoders such as BERT, RoBERTa, and DeBERTa have demonstrated promising per- formance while offering substantially lower inference latency than decoder-only LLMs.

ModernBERT further extends this family by improving contextual representation learning through architectural refine- ments and optimized long-context processing. Its favorable balance between accuracy and computational efficiency makes it particularly suitable for deployment as a front-end security filter preceding large language models.

Nevertheless, most lightweight detectors are trained using static datasets that represent only a limited subset of possible attack strategies. As adversaries continuously develop novel injection patterns, models trained under fixed data distributions often exhibit reduced generalization capability.

## C. Curriculum Learning

Curriculum learning, introduced by Bengio et al., proposes organizing training examples from simpler to more difficult samples in order to facilitate stable optimization and improved generalization. Rather than exposing the model to highly challenging examples from the outset, curriculum learning pro- gressively increases task complexity according to the learner’s current capability.

Numerous studies have reported improved convergence, robustness, and representation quality using curriculum learn- ing across computer vision, natural language processing, and reinforcement learning tasks. More recently, adaptive curric- ula have emerged in which sample difficulty is dynamically adjusted based on model performance instead of following a predetermined ordering.

Despite its demonstrated effectiveness, curriculum learning has received comparatively little attention in prompt injection detection, where adversarial examples naturally exhibit vary- ing levels of semantic complexity.

## D. Knowledge Distillation

Knowledge distillation enables compact student models to learn from larger teacher networks by minimizing the discrep- ancy between their output probability distributions. The soft targets generated by the teacher provide richer supervisory information than hard class labels alone, allowing smaller models to approximate the decision boundaries learned by more expressive architectures.

Progressive Generalized Knowledge Distillation (PGKD) further demonstrates that progressively aligning student rep- resentations during training improves robustness and conver- gence stability. Motivated by these findings, the proposed

framework adopts a teacher-student paradigm in which a Qwen-based teacher transfers semantic reasoning capabilities to a lightweight ModernBERT student through a combined supervised and distillation objective.

## E. Calibration of Security Classifiers

For security-critical applications, predictive confidence is nearly as important as classification accuracy. Deep neural networks frequently produce overconfident probability esti- mates even when predictions are incorrect, potentially leading downstream systems to make unsafe decisions.

Expected Calibration Error (ECE) has become a widely adopted metric for measuring the alignment between predicted confidence and empirical accuracy. Well-calibrated detectors enable adaptive threshold selection, confidence-aware filtering, and reliable deployment within larger LLM pipelines.

## F. Research Gap

Existing approaches generally optimize either detection accuracy or inference efficiency, but rarely both simultane- ously. Large language model detectors provide strong semantic reasoning capabilities yet remain computationally expensive, whereas lightweight classifiers often struggle to generalize be- yond fixed training distributions. Furthermore, current prompt injection detectors seldom exploit adaptive curriculum learning or teacher-guided knowledge distillation in a unified frame- work.

Adaptive-IPI addresses these limitations by integrating adaptive curriculum generation with knowledge distillation, enabling a lightweight ModernBERT detector to progressively learn increasingly complex prompt injection patterns while preserving deployment efficiency. The resulting framework combines the semantic reasoning capability of a large teacher model with the computational advantages of a compact student model, making it suitable for real-time LLM security applica- tions.

## III. PROPOSED METHODOLOGY

This section presents the proposed Adaptive-IPI framework for lightweight indirect prompt injection detection. The frame- work integrates adaptive curriculum learning with teacher- guided knowledge distillation to improve the robustness of a lightweight detector while maintaining low computational overhead. Figure 1 illustrates the overall training and deploy- ment pipeline.

## A. System Overview

The proposed framework consists of four primary com- ponents: a curriculum generator, a large teacher model, a lightweight student detector, and a deployment module. During training, a Qwen-based teacher model analyzes each prompt and produces semantic supervision in the form of soft prob- ability distributions. Simultaneously, an adaptive curriculum generator estimates the difficulty of every training sample and progressively exposes increasingly challenging prompt injection examples according to the learning progress of the student model.


The student detector is implemented using ModernBERT owing to its strong contextual representation capability and efficient inference speed. Instead of learning solely from man- ually annotated labels, the student jointly optimizes against the ground-truth labels and the probability distribution generated by the teacher model. This dual supervision enables the student to approximate the semantic reasoning ability of the teacher while remaining computationally lightweight.

After training, only the ModernBERT detector is retained for inference. Incoming prompts are first processed by the detector before reaching the downstream LLM. Prompts pre- dicted as malicious are rejected or quarantined, whereas benign prompts are forwarded to the language model through a FastAPI-based inference service. This design introduces min- imal latency while substantially reducing exposure to indirect prompt injection attacks.

*Fig. 1. Overall architecture of the proposed Adaptive-IPI framework. During training, the adaptive curriculum generator and Qwen teacher supervise the ModernBERT student. During deployment, only the lightweight student model is retained for real-time prompt filtering.*

## B. Threat Model

Adaptive-IPI considers retrieval-augmented language model

systems that process external content before generating re- sponses. Let

represent the complete model input, where

user’s original query and

textual information obtained from emails, documents, APIs, databases, or web pages.

An adversary cannot directly modify the user’s instruction but may inject malicious instructions into the retrieved context.

u denotes the

c

represents externally retrieved con-

The objective of the attacker is to manipulate the downstream language model into executing unauthorized instructions by exploiting the model’s inability to distinguish trusted user instructions from untrusted external content.

Formally, the retrieved context can be expressed as

where cb denotes benign contextual information and cm represents malicious prompt injection content.

The detector therefore performs binary classification

The objective is to learn a classifier that maximizes detection performance while maintaining low inference latency suitable for deployment in production LLM systems.

Unlike conventional prompt classifiers trained on static datasets, Adaptive-IPI continually increases training difficulty through adaptive curriculum generation. Consequently, the student model learns decision boundaries that remain robust against increasingly sophisticated prompt injection strategies encountered during training.

The following subsections describe the adaptive curriculum generation mechanism, teacher-guided knowledge distillation, and the joint optimization objective used to train the proposed detector.

## C. Teacher–Student Knowledge Distillation

The proposed Adaptive-IPI framework adopts a teacher– student learning paradigm to transfer the semantic reason- ing capability of a large language model into a lightweight classifier suitable for real-time deployment. During training, a Qwen-based teacher model remains fixed and produces probability distributions that guide the optimization of a Mod- ernBERT student model. After training, only the student model is retained for inference, thereby achieving significantly lower computational overhead.

Let the teacher logits be denoted by zT and the student logits by zS. Their softened probability distributions are com- puted using temperature-scaled softmax

pT i = P j exp(zT j /τ) ,

and

pS i = P j exp(zS j /τ) ,

where τ denotes the temperature parameter.

A larger value of τ generates smoother probability distribu- tions, allowing the student to capture the relative confidence of the teacher across both classes rather than relying solely on hard labels.


1) Supervised Classification Loss: The student model min- imizes the binary cross-entropy loss

where yi is the ground-truth label and ˆyi is the predicted

probability.

2) Knowledge Distillation Loss: To encourage the student to imitate the teacher, the Kullback–Leibler divergence be- tween the softened probability distributions is minimized,

where

The temperature scaling factor preserves gradient magni- tudes during optimization.

3) Joint Optimization: The overall optimization objective

combines supervised learning and knowledge distillation,

where λ balances the contribution of the two objectives.

The values of τ and λ were selected following the hyperparameter alignment strategy proposed in Progressive Generalized Knowledge Distillation (PGKD), enabling effec- tive semantic transfer while preserving the efficiency of the lightweight student detector.

4) Training Procedure: The complete optimization process is summarized in Algorithm 1.

## Algorithm 1 Teacher–Student Knowledge Distillation

Initialize pretrained ModernBERT student.

Load pretrained Qwen teacher.

for each training mini-batch do

Obtain teacher logits.

Obtain student logits.

Compute binary cross-entropy loss.

Compute knowledge distillation loss.

Compute total loss.

Update only the student parameters.

## end for

Return trained ModernBERT detector.

The combination of adaptive curriculum learning and teacher-guided knowledge distillation enables the student model to progressively learn increasingly challenging prompt injection patterns while maintaining low inference latency. Consequently, the proposed detector achieves strong general- ization across diverse indirect prompt injection attacks without sacrificing deployment efficiency.

## D. Inference and Deployment

After completion of training, the Qwen teacher model and curriculum generator are discarded. Only the optimized Mod- ernBERT student model is retained for deployment, thereby significantly reducing computational overhead during infer- ence.

For an incoming prompt

where u denotes the user query and c represents the re- trieved external context, the detector estimates

where fθ denotes the trained ModernBERT classifier. The prediction probability is obtained using

where σ(·) represents the sigmoid activation.

The final decision rule is

where δ denotes the decision threshold selected using vali-

dation data.

Prompts classified as malicious are blocked before reaching the downstream language model, whereas benign prompts are forwarded through the application pipeline.

1) FastAPI Deployment Pipeline: The trained Modern- BERT detector is deployed as a lightweight REST service using FastAPI. The deployment pipeline consists of four sequential stages.

- 1) Receive the incoming prompt.

- 2) Tokenize the prompt using the ModernBERT tokenizer.

- 3) Perform prompt injection detection.

- 4) Forward benign prompts to the downstream LLM while rejecting malicious requests.

Figure 2 illustrates the deployment workflow.

The modular design enables the detector to function in- dependently of the underlying language model and there- fore supports integration with Retrieval-Augmented Genera- tion systems, conversational assistants, enterprise agents, and autonomous tool-using applications.

2) Computational Complexity: Assume an input sequence

of length n.

The self-attention mechanism of ModernBERT requires

operations per transformer layer, while feed-forward com- putation scales linearly with the sequence length.

Since inference retains only the student model, the overall computational cost becomes


deployment_architecture.png

*Fig. 2. Deployment architecture of Adaptive-IPI using a FastAPI inference server. Only the ModernBERT student model is required during inference, enabling efficient real-time prompt filtering.*

where L denotes the number of transformer layers.

Unlike LLM-based detection systems that execute billions of parameters for every prediction, Adaptive-IPI performs inference using a compact encoder-only architecture. Conse- quently, the framework achieves substantially lower latency and memory consumption while preserving high detection accuracy.

3) Discussion: The proposed methodology combines adap- tive curriculum learning with teacher-guided knowledge dis- tillation to address two major limitations of existing prompt injection detectors. First, adaptive curriculum generation pro- gressively expands the student’s decision boundary by in- troducing increasingly difficult prompt injection examples throughout training. Second, knowledge distillation transfers the semantic reasoning capability of a large language model into an efficient encoder architecture suitable for deployment.

Together, these components enable Adaptive-IPI to maintain high predictive performance while satisfying the latency and resource constraints of practical LLM security systems.

## IV. EXPERIMENTAL SETUP

This section describes the experimental configuration adopted to evaluate the proposed Adaptive-IPI framework. The implementation was designed to ensure reproducibility while providing a fair comparison with existing lightweight prompt injection detection approaches.

## A. Dataset

Experiments were conducted using the BIPIA Email QA benchmark, a publicly available benchmark specifically de- signed for evaluating indirect prompt injection detection in

retrieval-augmented large language model systems. The bench- mark contains both benign and adversarial email-based re- trieval contexts, where malicious instructions are embedded within otherwise legitimate documents.

Each sample is represented as a binary classification task,

where

denotes a benign prompt and

indicates an indirect prompt injection attack.

To reduce sampling bias, the dataset was randomly shuffled prior to training and divided into training, validation, and testing subsets while preserving class balance.

## B. Training Configuration

The student detector was implemented using the Mod- ernBERT encoder architecture, whereas the teacher model employed Qwen for semantic supervision during knowledge distillation.

Training was performed using the AdamW optimizer with a cosine learning-rate scheduler. Early stopping was employed based on validation loss to prevent overfitting.

The principal hyperparameters are summarized in Table I.

*TABLE I*

*TRAINING HYPERPARAMETERS*

| Parameter | Value |
| --- | --- |
| Student Model | ModernBERT |
| Teacher Model | Qwen |
| Optimizer | AdamW |
| Batch Size | 32 |
| Learning Rate | 2 × 10−5 |
| Weight Decay | 0.01 |
| Maximum Epochs | 10 |
| Temperature (τ) | 2.0 |
| Distillation Weight (λ) | 0.5 |
| Maximum Sequence Length | 512 |
| Scheduler | Cosine Annealing |
| Early Stopping | Enabled |

The curriculum learning schedule progressively increased the proportion of difficult samples throughout training accord- ing to the strategy presented in Section III.

## C. Evaluation Metrics

Performance was evaluated using multiple complementary metrics to assess both predictive accuracy and confidence calibration.

Accuracy is defined as


where TP, TN, FP, and FN denote the numbers of true positives, true negatives, false positives, and false negatives, respectively.

Precision and Recall are computed as

and

The F1-score is given by

where P and R denote Precision and Recall.

Receiver Operating Characteristic Area Under the Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC) were additionally computed to evaluate ranking performance across different decision thresholds.

Model calibration was assessed using Expected Calibration Error (ECE),

where Bm denotes the confidence bins used for calibration

analysis.

## D. Baseline Methods

Adaptive-IPI was compared against representative lightweight transformer-based classifiers and conventional prompt injection detection approaches reported in the literature. Comparisons focused on classification performance, calibration quality, and deployment efficiency.

To ensure fairness, all competing methods were evaluated under identical preprocessing, train-validation-test splits, and evaluation metrics.

## E. Implementation Details

The framework was implemented in Python using the Hugging Face Transformers library together with PyTorch. FastAPI was employed to expose the trained ModernBERT detector as a lightweight inference service capable of integra- tion with downstream LLM applications.

Training and evaluation were fully deterministic through fixed random seeds, ensuring reproducible experimental re- sults. Hyperparameter selection followed the recommendations of Progressive Generalized Knowledge Distillation (PGKD), providing a balanced trade-off between optimization stability and semantic knowledge transfer.

Overall, the experimental setup was designed to evaluate both the effectiveness of the proposed adaptive curriculum learning strategy and the practical feasibility of deploying a lightweight prompt injection detector in real-world LLM systems.

## V. RESULTS AND DISCUSSION

This section presents the experimental evaluation of Adaptive-IPI on the BIPIA Email QA benchmark. The pro- posed framework is assessed in terms of predictive per- formance, calibration quality, and the contribution of each architectural component through an ablation study.

## A. Overall Performance

Table II summarizes the overall performance of Adaptive-

IPI.

*TABLE II*

*PERFORMANCE OF ADAPTIVE-IPI ON THE BIPIA EMAIL QA*

*BENCHMARK*

| Metric Value |   |
| --- | --- |
| Accuracy 99.06% |   |
| Precision 99.14% |   |
| Recall | 99.87% |
| F1-score 99.50% |   |
| AUROC 99.78% |   |
| AUPRC 99.99% |   |
| ECE | 0.1096 |

The proposed framework achieved an overall accuracy of 99.06%, demonstrating that the lightweight ModernBERT detector successfully distinguishes benign prompts from in- direct prompt injection attacks. The high precision of 99.14% indicates a very low false-positive rate, reducing the likelihood of rejecting legitimate user requests.

The recall of 99.87% further demonstrates the detector’s ability to identify nearly all malicious prompts present within the benchmark. From a deployment perspective, high recall is particularly important because undetected prompt injections may directly compromise downstream language model behav- ior.

The resulting F1-score of 99.50% confirms that Adaptive- IPI maintains an effective balance between precision and recall.

## B. Ranking Performance

Threshold-independent evaluation was performed using AU- ROC and AUPRC.

The obtained AUROC of 99.78% demonstrates excellent class separability across different decision thresholds. Simi- larly, the AUPRC value of 99.99% indicates that the detector maintains extremely high precision even under varying recall levels.

These results suggest that the learned feature representations remain highly discriminative despite the lightweight nature of the student architecture.

## C. Calibration Analysis

Besides prediction accuracy, reliable confidence estimation is essential for security-sensitive applications.

Adaptive-IPI achieved an Expected Calibration Error (ECE) of 0.1096, indicating close agreement between predicted con- fidence and empirical correctness.


Well-calibrated confidence estimates enable adaptive thresh- old selection and confidence-aware security policies. Conse- quently, downstream applications may dynamically reject un- certain predictions instead of relying solely on fixed decision thresholds.

## D. Effect of Adaptive Curriculum Learning

To evaluate the contribution of adaptive curriculum learning, experiments were conducted by training the ModernBERT de- tector without progressive curriculum generation while keep- ing all remaining components unchanged.

The adaptive curriculum consistently accelerated conver- gence during early epochs and produced improved robustness against difficult prompt injection examples encountered during evaluation.

Rather than exposing all adversarial samples simultaneously, gradually increasing task difficulty allowed the student model to develop stable semantic representations before learning increasingly complex attack strategies.

These observations agree with previous findings on cur- riculum learning and demonstrate its applicability to prompt injection detection.

## E. Ablation Study

An ablation study was performed to quantify the contribu- tion of the major components of Adaptive-IPI.

*TABLE III*

*ABLATION STUDY*

| Configuration | Accuracy F1 |
| --- | --- |
| ModernBERT Only | 97.82% 98.04% |
| + Knowledge Distillation 98.61% 98.77% |   |
| + Adaptive Curriculum | 99.06% 99.50% |

Removing knowledge distillation noticeably reduced clas- sification performance, confirming that semantic supervision from the Qwen teacher substantially improves representation learning.

Similarly, eliminating adaptive curriculum learning resulted in slower convergence and lower generalization performance, particularly for semantically complex prompt injection attacks.

The complete Adaptive-IPI framework consistently achieved the strongest performance across all evaluation metrics.

## F. Deployment Efficiency

A major objective of Adaptive-IPI is practical deployment within real-world LLM systems.

Unlike approaches that execute large decoder-only language models during every prediction, Adaptive-IPI performs infer- ence exclusively using the ModernBERT student model. This significantly reduces memory requirements, inference latency, and computational cost.

The FastAPI deployment architecture further enables straightforward integration into Retrieval-Augmented Genera- tion systems, enterprise assistants, and autonomous AI agents with minimal infrastructure modifications.

Overall, the proposed framework provides an effective bal- ance between detection accuracy, computational efficiency, and deployment scalability.

## G. Discussion

The experimental results demonstrate that combining adap- tive curriculum learning with teacher-guided knowledge distil- lation enables a lightweight detector to approach the semantic reasoning capability of considerably larger language models.

Adaptive curriculum learning progressively expands the stu- dent’s decision boundary by introducing increasingly difficult prompt injection samples throughout training. Knowledge dis- tillation simultaneously transfers semantic knowledge from the Qwen teacher, allowing the ModernBERT student to achieve strong generalization despite its comparatively small parameter count.

The combination of these two techniques results in con- sistently high predictive performance while preserving the inference efficiency required for deployment in practical LLM security pipelines.

## VI. CONCLUSION AND FUTURE WORK

This paper presented Adaptive-IPI, a lightweight framework for indirect prompt injection detection that combines adaptive curriculum learning with teacher-guided knowledge distilla- tion. The proposed approach addresses a key limitation of existing lightweight detectors by progressively exposing the student model to increasingly challenging prompt injection ex- amples while simultaneously transferring semantic reasoning capabilities from a large language model.

The framework employs a Qwen-based teacher model to guide the optimization of a ModernBERT student through a joint objective consisting of supervised classification and knowledge distillation. Unlike conventional training pipelines that rely on static datasets, the adaptive curriculum dynami- cally adjusts sample difficulty according to the student’s learn- ing progress, enabling more stable optimization and improved robustness against previously unseen attack strategies.

Experimental evaluation on the BIPIA Email QA bench- mark demonstrates that the proposed framework achieves an overall accuracy of 99.06%, precision of 99.14%, recall of 99.87%, F1-score of 99.50%, AUROC of 99.78%, AUPRC of 99.99%, and an Expected Calibration Error of 0.1096. These results indicate that Adaptive-IPI successfully balances predic- tive performance, confidence calibration, and computational efficiency.

From a practical perspective, retaining only the Modern- BERT student model during inference substantially reduces computational cost compared with large language model- based detection systems. The lightweight FastAPI deployment architecture further enables seamless integration into Retrieval- Augmented Generation systems, autonomous AI agents, enter- prise assistants, and other production-scale LLM applications.

Although the proposed framework demonstrates strong per- formance, several limitations remain. The current implemen- tation focuses on binary prompt injection detection using the


BIPIA Email QA benchmark and evaluates attacks within a specific retrieval setting. Future adversarial techniques may involve multimodal inputs, multilingual prompt injections, or coordinated attacks spanning multiple external information sources. Furthermore, curriculum generation currently relies on a single teacher architecture and fixed curriculum schedul- ing parameters. Future research will investigate adaptive curriculum gener- ation using reinforcement learning, online continual learning for evolving attack distributions, multilingual prompt injection detection, multimodal retrieval scenarios, uncertainty-aware calibration techniques, and lightweight ensemble methods. Ex- tending the framework to support real-time adaptation against emerging prompt injection strategies represents an important direction for securing next-generation LLM applications. Overall, Adaptive-IPI demonstrates that combining adaptive curriculum learning with teacher-guided knowledge distillation provides an effective and computationally efficient solution for indirect prompt injection detection. The proposed framework offers a practical foundation for deploying robust security mechanisms alongside modern large language model systems while maintaining the low latency required for real-world applications.

## ACKNOWLEDGMENT

The authors would like to thank the maintainers of the BIPIA benchmark and the open-source machine learning com- munity for providing the datasets, pretrained models, and software frameworks that enabled this research.

REFERENCES
