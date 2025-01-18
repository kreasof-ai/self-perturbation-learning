<h1 align="center">Self-Perturbation Learning for Mathematical Reasoning Verification</h1>

<h3 align="center"><a href="https://kreasof.my.id/" style="color: #2088FF;">Kreasof AI</a></h3>

**Abstract:**

This document outlines a proof-of-concept project for developing a robust verifier model for mathematical reasoning using a novel self-supervised technique called Self-Perturbation Learning (SPL). We leverage the publicly available FineMath dataset, a large-scale corpus of mathematical educational content comprising 34 billion tokens in `FineMath-3+` and 54 billion tokens in `FineMath-3+ with InfiMM-WebMath-3+`, and a modified ModernBERT architecture to create a model that can effectively assess the correctness and quality of mathematical text. By training the model to identify artificially introduced errors ("impostors") through a self-perturbation process guided by semantic embedding distances, we create a verifier that can identify subtle deviations in mathematical reasoning and context, paving the way for more reliable AI systems in math.

## **1. Introduction:**

The development of reliable AI systems capable of understanding and reasoning about complex domains like mathematics requires robust mechanisms for verifying information. Large language models (LLMs) have shown promise in generating mathematical content, but they often struggle with accuracy, consistency, and the detection of errors. This is evidenced by phenomena like "hallucinations" where the model generates mathematically unsound answers, making trust and reliability in such models particularly challenging for applications that require high accuracy. This project explores a novel approach to training a math-specific verifier model using Self-Perturbation Learning (SPL). We demonstrate the method's effectiveness by applying it to the FineMath dataset with a modified ModernBERT architecture. This allows us to leverage existing techniques to perform a self-supervised approach for the verification task with the goal to create a new method for creating a better verifier system for mathematical context.

## **2. Background and Related Work:**

-   **Challenges in Mathematical Reasoning:** Automated mathematical reasoning is a notoriously difficult challenge due to the symbolic and formal nature of the language. It requires a deep understanding of logical and mathematical principles beyond simple pattern recognition. Mathematical reasoning problems also are difficult since it requires rigorous step-by-step solutions which need to be evaluated not only based on the end results but also the steps.
-   **Limitations of Existing Methods:**  Existing approaches for verifying mathematical content are often either rule-based systems (that are brittle and difficult to maintain) or rely heavily on human labeling (which is expensive and time-consuming). These methods are limited in their ability to generalize to diverse mathematical content and cannot accurately assess the subtle nuances of mathematical argumentation.
-   **Self-Supervised Learning:** Self-supervised learning methods have shown immense potential in leveraging unlabeled data to create robust representations. By creating a supervisory signal from the inherent structure of the data, it avoids the need for expensive and cumbersome manual labeling which can be biased and might not generalize well.
-   **ModernBERT:**  The ModernBERT architecture provides a state-of-the-art (SOTA) performance for text processing, with a strong focus on efficiency, especially in longer sequence contexts, which is needed for processing mathematical text and the solutions steps. It improves upon the original BERT architecture by incorporating recent advancements including GeGLU activations, RoPE embeddings, and alternating local and global attention. This model has a native sequence length of 8,192 tokens and trained on 2 trillion tokens of data including code, math and general text and makes it a great base model for fine-tuning into our SPL based math verifier model.
-   **FineMath Dataset:** The FineMath dataset provides a large-scale corpus of math educational content extracted from Common Crawl using Llama-3.1-70B-Instruct to filter high-quality mathematical content and step-by-step problem-solving solutions, making it an ideal candidate for our task to train a math verifier model, it consists of two version, FineMath-3+, and FineMath-4+, and also includes a filtered English text-only portion of the InfiMM-WebMath dataset.

## **3. Self-Perturbation Learning (SPL) Methodology:**

-   **Core Principles:** SPL trains a critic model by identifying intentionally introduced errors ("impostor elements") in data. These errors are generated through a self-perturbation process where words, pixels, sounds, or other modality specific elements are replaced with alternative ones, forcing the model to learn a robust understanding of what constitutes correct data and to identify deviations from that.
-   **Impostor Word Sampling:**  We generate "impostor" words by first selecting the top-K most frequent words from the FineMath dataset, calculating their embedding (using the ModernBERT model), and then sampling nearby words using the Cosine similarity as a distance measure in the embedding space. This method introduces context-aware "impostor" words which can be adjusted to create different levels of challenges and make the training task more meaningful for the model.
-   **Embedding Distance:** To allow the critic model to learn a more nuanced understanding of errors, we utilize the distance between original and substituted word embeddings, where lower distances generate harder "impostors" and larger distances creates easier "impostors". The critic can either be trained to predict the distance or to use it as a training signal to learn a relative score of deviation.
-   **Training Objective:** The SPL method is used for pretraining, and can further be fine-tuned with an additional downstream training objectives to further improve the model performance, using either a binary cross-entropy loss (for a binary "impostor" detection) or a mean-squared error loss function for predicting the distance between embeddings or a regression loss function for predicting the overall score.
-   **Adaptability:** It's important to highlight the adaptability of the technique, where SPL can be implemented on a wide range of data types such as text, image, audio and video by adjusting the type of perturbation applied to the original data.

## **4. Implementation of Math Verifier using ModernBERT:**

-   **Choice of Base Model:** ModernBERT was chosen due to its strong performance on text processing tasks, its efficient architecture and support of longer context length, which is suitable for complex math text, making it a great base for our verifier model.
-   **Data Preparation:** The FineMath dataset is downloaded through the Hugging Face datasets library.  LaTeX and Markdown formatting is kept in the data, and not removed, to allow the model to learn how to process these mathematical notations. Only the `text` field will be used in this experiment for the text processing, while the `score` field is used later for downstream evaluations, and the URL and meta-data can be kept if necessary for more experiments.
*  **SPL Data Creation Process:** To create the SPL training data, we first construct a list of top 10000 most frequent words in the FineMath dataset. Then, from the `text` field of each dataset entry, we randomly sample 15% of the words to be substituted with the "impostor" words, where the difficulty of each "impostor" is also randomly sampled to create variations during training. The difficulty levels are based on the top 10 nearest words for "Hard", top 10 to 50 for "Medium", and 50 to 100 for "Easy" levels.
-   **Model Architecture Adaptations:** The ModernBERT model is used without architectural modification. We use the embeddings and transformer layers to get the contextualized representation, and then add a pooling layer to summarize each sequence and then feed it into a single linear layer to generate an output of either a single scalar value for regression or a binary value for binary cross-entropy.

## **5. Experimental Setup:**

-   **Dataset Details:** We will use both `finemath-3plus` (34B tokens, 21.4M documents) and `finemath-4plus` (9.6B tokens, 6.7M documents of higher quality) datasets provided by Hugging Face datasets library for training and evaluations. We split the datasets randomly into 80% for training, 10% for validation, and 10% for testing.
-   **Implementation Details:** All experiments will be conducted on an NVIDIA A100 GPU, using PyTorch and the Hugging Face transformers library.
-   **Evaluation Metrics:** We will evaluate the pre-trained SPL critic with the following metrics:
    -   **Impostor Detection Accuracy:** To measure the accuracy of detecting inserted "impostor" words.
    -   **Correlation with Human Judgments:** The correlation between the critic’s evaluation scores and human evaluation of text quality for the subset of generated text samples.
    -   **Downstream performance:** We use datasets from the FineMath paper such as GSM8k and MATH to test and evaluate our model's performance on actual math-related tasks.
-   **Baseline Models:** We will also include a comparison with a randomly initialized model to see if there is actual improvement in the performance and a smaller version of the model that is not ModernBERT to test the effectiveness of architectural updates in the new model. We also plan to compare the result of SPL with a model trained with the traditional method, using the generator's cross-entropy loss.

## **6. Results and Discussion:**

-   **Performance Analysis:** This section will present the quantitative results of the experiments, such as the accuracy of identifying "impostor" words, the correlation with human judgments and downstream task scores, to demonstrate the performance of the SPL based math verifier model.
-   **Qualitative Analysis:** This section will provide detailed examples of how the verifier correctly flags the mathematical errors or anomalies and will compare the results in the scenarios with and without the context being considered by the model.
-   **Comparative Analysis:** We will also compare the results of the models trained with the SPL method with the baseline models, such as random and other architectures.
-   **Ablation Study:** This section will discuss the impact of using different difficulty settings on SPL training data, the effect of incorporating different types of embeddings and also testing out different loss function in the results.
-   **Analysis of Model Behaviour:** We analyze the behaviour of our model in different scenarios, focusing on explaining when the model can perform well, and when the performance is not as good, and what are the key reasons for it.

## **7. Conclusion and Future Work:**

-   **Summary of Findings:** Summarize the key findings, highlighting the success in the development of a proof-of-concept math verifier model trained using the SPL method. We will summarize whether SPL successfully improves the performance of verifier tasks in the mathematical domain.
-   **Contributions:** This work shows the effectiveness of using the SPL method for creating a strong verifier model that does not rely on human labels or mimicking the generator objective, and provides the methodology of developing a new method to generate data for SPL based training approach.
-   **Limitations:** Acknowledge potential limitations, such as reliance on embedding quality, the limitations of the specific perturbations applied and the scope of mathematical content covered in the dataset.
-   **Future Research Directions:**
    -   Explore different variations of SPL using different methods of word selection, and perturbation techniques, and analyze the impact of their contribution.
    -   Adapt and implement the SPL for other data modalities to create a "universal verifier model."
    -   Incorporate other signals, such as human feedback or symbolic reasoning techniques, to improve the verifier's ability to assess mathematical content.
    -   Use the SPL trained models to act as a safety and reliability guard for generative math models.

-   **Potential Impact:** This research has the potential to create more reliable and robust AI systems for mathematical reasoning, providing a valuable tool for math education, research, and other related applications.

## **8. References:**

```
@misc{lozhkov2024finemath,  
    author       = { Lozhkov, Anton and Ben Allal, Loubna and Bakouch, Elie and von Werra, Leandro and Wolf, Thomas },  
    title        = { FineMath: the Finest Collection of Mathematical Content }, 
    year         = 2024,  
    url          = { https://huggingface.co/datasets/HuggingFaceTB/finemath },  
    doi          = { 10.57967/hf/3847 },
    publisher    = { Hugging Face }
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