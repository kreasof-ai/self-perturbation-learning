<h1 align="center">Self-Perturbation Learning for Mathematical Reasoning Verification: "2 truth and a lie" as machine learning algorithms</h1>

<h3 align="center"><a href="https://kreasof.my.id/" style="color: #2088FF;">Kreasof AI</a></h3>

**Abstract:**

This document outlines a proof-of-concept project for developing a robust verifier model for mathematical reasoning using a novel self-supervised technique called Self-Perturbation Learning (SPL). We leverage the publicly available AutoMathText dataset, a large-scale corpus of mathematical content comprising 200 GB of mathematical texts in `math-ai/AutoMathText`, and a modified ModernBERT architecture to create a model that can effectively assess the correctness and quality of mathematical text. By training the model to identify artificially introduced errors ("impostors") through a self-perturbation process guided by semantic embedding distances, we create a verifier that can identify subtle deviations in mathematical reasoning and context, paving the way for more reliable AI systems in math.

## **1. Introduction:**

The development of reliable AI systems capable of understanding and reasoning about complex domains like mathematics requires robust mechanisms for verifying information. Large language models (LLMs) have shown promise in generating mathematical content, but they often struggle with accuracy, consistency, and the detection of errors. This is evidenced by phenomena like "hallucinations" where the model generates mathematically unsound answers, making trust and reliability in such models particularly challenging for applications that require high accuracy. This project explores a novel approach to training a math-specific verifier model using Self-Perturbation Learning (SPL). We demonstrate the method's effectiveness by applying it to the AutoMathText dataset with a modified ModernBERT architecture. This allows us to leverage existing techniques to perform a self-supervised approach for the verification task with the goal to create a new method for creating a better verifier system for mathematical context.

## **2. Background and Related Work:**

-   **Challenges in Mathematical Reasoning:** Automated mathematical reasoning is a notoriously difficult challenge due to the symbolic and formal nature of the language. It requires a deep understanding of logical and mathematical principles beyond simple pattern recognition. Mathematical reasoning problems also are difficult since it requires rigorous step-by-step solutions which need to be evaluated not only based on the end results but also the steps.
-   **Limitations of Existing Methods:**  Existing approaches for verifying mathematical content are often either rule-based systems (that are brittle and difficult to maintain) or rely heavily on human labeling (which is expensive and time-consuming). These methods are limited in their ability to generalize to diverse mathematical content and cannot accurately assess the subtle nuances of mathematical argumentation.
-   **Self-Supervised Learning:** Self-supervised learning methods have shown immense potential in leveraging unlabeled data to create robust representations. By creating a supervisory signal from the inherent structure of the data, it avoids the need for expensive and cumbersome manual labeling which can be biased and might not generalize well.
-   **ModernBERT:**  The ModernBERT architecture provides a state-of-the-art (SOTA) performance for text processing, with a strong focus on efficiency, especially in longer sequence contexts, which is needed for processing mathematical text and the solutions steps. It improves upon the original BERT architecture by incorporating recent advancements including GeGLU activations, RoPE embeddings, and alternating local and global attention. This model has a native sequence length of 8,192 tokens and trained on 2 trillion tokens of data including code, math and general text and makes it a great base model for fine-tuning into our SPL based math verifier model.
-   **AutoMathText Dataset:** AutoMathText is an extensive and carefully curated dataset encompassing around 200 GB of mathematical texts. It's a compilation sourced from a diverse range of platforms including various websites, arXiv, and GitHub (OpenWebMath, RedPajama, Algebraic Stack). This rich repository has been autonomously selected (labeled) by the state-of-the-art open-source language model, Qwen-72B. Each piece of content in the dataset is assigned a score lm_q1q2_score within the range of [0, 1], reflecting its relevance, quality and educational value in the context of mathematical intelligence.

## **3. Self-Perturbation Learning (SPL) Methodology:**

-   **Core Principles:** SPL trains a critic model by identifying intentionally introduced errors ("impostor elements") in data. These errors are generated through a self-perturbation process where words, pixels, sounds, or other modality specific elements are replaced with alternative ones, forcing the model to learn a robust understanding of what constitutes correct data and to identify deviations from that.
-   **Impostor Word Sampling:**  We generate "impostor" words by first selecting the top-K most frequent words from the AutoMathText dataset, calculating their embedding (using the ModernBERT model), and then sampling nearby words using the Cosine similarity as a distance measure in the embedding space. This method introduces context-aware "impostor" words which can be adjusted to create different levels of challenges and make the training task more meaningful for the model.
-   **Embedding Distance:** To allow the critic model to learn a more nuanced understanding of errors, we utilize the distance between original and substituted word embeddings, where lower distances generate harder "impostors" and larger distances creates easier "impostors". The critic can either be trained to predict the distance or to use it as a training signal to learn a relative score of deviation.
-   **Training Objective:** The SPL method is used for pretraining, and can further be fine-tuned with an additional downstream training objectives to further improve the model performance, using either a binary cross-entropy loss (for a binary "impostor" detection) or a mean-squared error loss function for predicting the distance between embeddings or a regression loss function for predicting the overall score.
-   **Adaptability:** It's important to highlight the adaptability of the technique, where SPL can be implemented on a wide range of data types such as text, image, audio and video by adjusting the type of perturbation applied to the original data.

## **4. Implementation of Math Verifier using ModernBERT:**

-   **Choice of Base Model:** ModernBERT was chosen due to its strong performance on text processing tasks, its efficient architecture and support of longer context length, which is suitable for complex math text, making it a great base for our verifier model.
-   **Data Preparation:** The AutoMathText dataset is downloaded through the Hugging Face datasets library.  LaTeX and Markdown formatting is kept in the data, and not removed, to allow the model to learn how to process these mathematical notations. Only the `text` field will be used in this experiment for the text processing.
*  **SPL Data Creation Process:** To create the SPL training data, we first construct a list of top-K most frequent words in the AutoMathText dataset. Then, from the `text` field of each dataset entry, we randomly sample 40% of the words to be substituted with the "impostor" words, where the difficulty of each "impostor" is also randomly sampled to create variations during training. The difficulty levels are based on the top 10 nearest words for "Hard", top 10 to 50 for "Medium", and 50 to 100 for "Easy" levels.
-   **Model Architecture Adaptations:** The ModernBERT model is used without architectural modification. We use the embeddings and transformer layers to get the contextualized representation, and then add a pooling layer to summarize each sequence and then feed it into a single linear layer to generate an output of either a single scalar value for regression or a binary value for binary cross-entropy.

## **5. Experimental Setup:**

-   **Dataset Details:** We utilize a cleaned subset of the `math-ai/AutoMathText` dataset (specifically, the `web-0.80-to-1.00` subset), which offers a higher-quality, though smaller, corpus of mathematical educational content compared to the `FineMath` dataset used in initial experiments. For this experiment, we use a training set of 30,000 data points from this cleaned subset.
-   **Implementation Details:** Experiments are conducted using an NVIDIA GPU. We train the ModernBERT-base model using the Self-Perturbation Learning (SPL) methodology, with a batch size of 16, AdamW optimizer, a learning rate of 1e-5, weight decay of 1e-2, and a linear warmup with cosine decay learning rate schedule over one epoch.
-   **Sequence Quality Scoring (Revised):** We revise the sequence quality scoring method to enhance interpretability. We first calculate the negative log-likelihood of each sequence based on the token-level "impostor" probabilities. Then, we negate the negative log-likelihood and apply min-max scaling to map the scores to a more intuitive range. In this experiment, we use a min-max range of (-1, 0) for scaling the *negated* negative log-likelihood, resulting in revised sequence quality scores that range from negative values (for irrelevant or low-quality text) to values approaching 1.0 (for high-quality mathematical text).
-   **Evaluation Metrics:** We continue to evaluate the model using hardcoded examples, focusing on qualitative analysis of the revised sequence quality scores and their alignment with expected quality levels. We assess the model's ability to:
    -   Distinguish between relevant and irrelevant sequences.
    -   Discern different levels of quality within mathematical text (high-quality explanations, logical fallacies, word problems).
    -   Produce interpretable and nuanced quality scores.



## **6. Results and Discussion:**

-   **Improved Score Interpretability:** The revised min-max scaling and negation of the negative log-likelihood result in sequence quality scores that are significantly more interpretable and aligned with human intuition. Scores now range from negative values for irrelevant sequences to values approaching 1.0 for high-quality mathematical text.
-   **Clearer Quality Differentiation:** The verifier model trained on 30K cleaned data points demonstrates an enhanced ability to differentiate between various levels of text quality:
    -   **Irrelevant Sequences (e.g., "Cat sat on the mat"):** Consistently receive negative sequence quality scores (e.g., -153.23), clearly indicating their irrelevance to mathematical content.
    -   **High-Quality Math Explanations:** Achieve near-perfect revised scores (e.g., 0.9985 for correct algebraic simplification), reflecting high confidence in their quality.
    -   **Logical Fallacies:** Are now more distinctly scored in a "medium" quality range (e.g., 0.360-0.451), showing improved detection of logical flaws compared to earlier experiments.
    -   **Word Problems (Correct vs. Incorrect):** Revised scores show a more pronounced difference between correct (0.801) and incorrect (0.554) word problem answers, indicating enhanced sensitivity to solution correctness.

-   **Enhanced Qualitative Alignment:** The revised sequence quality scores now provide a more nuanced and intuitive measure of text quality, aligning better with human qualitative assessments across different types of examples. The expanded score range and negative values for irrelevant sequences provide a clearer and more informative signal from the verifier model.

-   **Analysis of Model Behavior:** The model demonstrates a refined ability to:
    -   Identify subtle inconsistencies and deviations in mathematical reasoning.
    -   Distinguish between different levels of quality within the mathematical domain.
    -   Robustly classify out-of-domain or irrelevant text with negative quality scores.

## **7. Conclusion and Future Work:**

-   **Summary of Findings:**  Experiments with 30K cleaned data points from `math-ai/AutoMathText` and revised min-max scaling demonstrate significant improvements in the performance and interpretability of the SPL-trained math verifier. The model now produces a more nuanced and intuitively scaled sequence quality score that effectively distinguishes between high-quality, flawed, and irrelevant text examples.
-   **Contributions:** This iteration further refines the SPL methodology and demonstrates its effectiveness in creating a highly interpretable and performant verifier model for mathematical reasoning. The revised sequence quality scoring method enhances the practical usability of the verifier by providing a more intuitive 0-1 quality metric with negative values for out-of-domain content.
-   **Limitations:** While the model shows strong qualitative performance, further quantitative evaluation on larger and more diverse test sets is needed to rigorously assess its accuracy, robustness, and calibration. Further investigation into the verifier's sensitivity to subtle logical fallacies and nuanced quality differences is also warranted.
-   **Future Research Directions:**
    -   Conduct comprehensive quantitative evaluations using metrics such as correlation with human judgments and AUC curves on larger and more diverse test sets.
    -   Scale up training to larger datasets (e.g., 500K+ data points) and larger models (e.g., ModernBERT-large) to further improve performance and robustness.
    -   Refine the "impostor" generation process to create more challenging and targeted errors, particularly for logical fallacies and subtle mathematical mistakes.
    -   Explore the use of the revised sequence quality score in downstream applications, such as LLM output verification and automated math tutoring systems.

## **8. References:**

```
@article{zhang2024automathtext,
      title={Autonomous Data Selection with Language Models for Mathematical Texts},
      author={Zhang, Yifan and Luo, Yifan and Yuan, Yang and Yao, Andrew Chi-Chih},
      journal={arXiv preprint arXiv:2402.07625},
      year={2024},
}
```

```
@misc{modernbert,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference}, 
      author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
      year={2024},
      eprint={2412.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13663}, 
}
```