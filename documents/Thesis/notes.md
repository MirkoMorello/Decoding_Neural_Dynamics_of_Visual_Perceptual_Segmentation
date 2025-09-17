### The Overarching Narrative

1.  **Replicate & Improve:** Start with the SoTA (DeepGaze III).
2.  **Hypothesize & Test:** Replace the backbone with a ViT (DinoGaze) based on a hypothesis about implicit segmentation.
3.  **Question & Isolate:** Realize the first success is ambiguous. Devise a method (SPADE) to isolate the variable of interest (explicit segmentation).
4.  **Innovate & Overcome:** Face a new problem (semantic coherence) and invent a solution (semantic painting).
5.  **Synthesize & Conclude:** Combine the best of all worlds (DinoGaze-SPADE) and propose a framework for the future.

This is the story we will tell across the chapters.

### The Optimal Thesis Structure

Based on your notes, here is a proposed structure that logically organizes your content. I've given each chapter a clear purpose.

**Chapter 1: Introduction**
*   (As we've already refined it) Sets the high-level stage.

**Chapter 2: State of the Art**
*   (As we've already refined it) Covers the necessary background, from the history of saliency to the probabilistic framework and the motivation from Vacher, Roth, and Brusco.

**Chapter 3: Experimental Methodology**
*   **Purpose:** To describe the *what* and *how* of your experiments. This is the place for the technical details of the data, the metrics, and the experimental setup. It's the "recipe book" for your research.
*   **Content:**
    *   **Section 3.1: Datasets.** A deep dive. Explain SALICON (mouse-tracking, "pseudofixations," large scale for pre-training) vs. MIT1003 (real eye-tracking, smaller, gold standard for scanpath evaluation). This is where you justify your multi-stage training protocol.
    *   **Section 3.2: Evaluation Framework.** Formally define the metrics (LL, IG, NSS, AUC) and the cross-validation protocol for MIT1003.
    *   **Section 3.3: Unsupervised Segmentation Mask Generation.** Detail the *process* of creating the masks here. Explain the methods (k-means on DINO features, SAM 16/64) and the rationale. This is a key methodological contribution.
    *   **Section 3.4: Multi-Stage Training Protocol.** Explicitly state and justify the three-stage training process (SALICON pre-train, MIT1003 spatial, MIT1003 scanpath). Explain that this is inherited from DeepGaze III and is a best practice for leveraging datasets of different types and sizes.

**Chapter 4: A Narrative of Model Development**
*   **Purpose:** This is your core "story" chapter. It's where you walk the reader through your research journey, introducing each model as a logical step in the narrative. **Crucially, you will describe the architecture and motivation for each model here, but you will only allude to the results, not present them in tables.** You can use phrases like "This model yielded a significant improvement, as will be detailed in Chapter 5," to build anticipation.
*   **Content:**
    *   **Section 4.1: The Baseline: DeepGaze III.** Briefly re-state its architecture as the starting point.
    *   **Section 4.2: Step 1 - The Implicit Segmentation Hypothesis: DinoGaze.** Introduce the model, explaining its architecture (ViT backbone) and motivating it with the ViT's emergent segmentation properties. State that it established a new baseline.
    *   **Section 4.3: Step 2 - Isolating Explicit Information: The SPADE Models.**
        *   `Subsection 4.3.1: The Rationale for SPADE.` Explain the "why" of SPADE (isolating variables, minimal complexity).
        *   `Subsection 4.3.2: An Initial Failure: DeepGaze-SPADE with Learned Embeddings.` Describe this model and explain *why* it was doomed to fail due to the semantic coherence problem. This is great storytelling.
        *   `Subsection 4.3.3: The Breakthrough: Semantic Painting with DeepGaze-SPADE.` Describe the architecture of your `v2` and `v3` models, explaining the semantic painting innovation. State that this approach successfully proved that explicit segmentation information provides a performance boost.
    *   **Section 4.4: The Final Framework: DinoGaze-SPADE.** Introduce the final model, combining the best backbone with the best information injection technique. Frame it not just as the best-performing model, but as a flexible framework for future research (the personalized segmentation idea).

**Chapter 5: Results and Analysis**
*   **Purpose:** This is the "evidence" chapter. Here, you present the data that supports the narrative you built in Chapter 4. The tables and figures belong here.
*   **Content:**
    *   **Section 5.1: Experiment 1 - The ViT Advantage.** Present the table comparing DeepGaze III and DinoGaze.
    *   **Section 5.2: Experiment 2 - The Impact of Explicit Segmentation.** Present the ablation study table comparing all the SPADE variants. This is the climax of your results.
    *   **Section 5.3: Experiment 3 - Generalization to Scanpath Prediction.** Present the table of results on the MIT1003 scanpath task.
    *   **Section 5.4: Qualitative Analysis.** Show the example saliency maps.

**Chapter 6: Implementation Details**
*   **Purpose:** This is the "ML-like" chapter you wanted. It's for the hardcore engineering details that are important for reproducibility but would bog down the main narrative. Some theses put this in an Appendix, but a dedicated chapter is perfectly fine and often preferred in ML-focused work.
*   **Content:**
    *   **Section 6.1: Data Pipeline and Preprocessing.** Talk about your PyTorch `Dataset` classes, LMDB caching, and shape-aware batching.
    *   **Section 6.2: Training and Experiment Orchestration.** Describe your training loop (DDP, AMP, etc.) and the orchestrator script with its YAML configs. This is where you showcase the rigor of your experimental setup.
    *   **Section 6.3: Codebase and Reproducibility.** Mention the model/data registry and any other software engineering best practices you followed.

**Chapter 7: Discussion and Future Work** (You called it "Conclusions" but "Discussion" is often better)
*   **Purpose:** To interpret your results in the context of the wider field and look forward.
*   **Content:**
    *   **Summary of Contributions:** A brief, high-level summary of your main findings.
    *   **Discussion:** Relate your results back to the literature (Koch, DeepGaze, Roth, Vacher). How does your work extend or challenge theirs? For example: "While Roth et al. showed the importance of computer-defined objects, our work provides a proof-of-concept for a framework that can leverage *perceptual* objects..."
    *   **Future Work:** This is where you expand on the personalized segmentation maps idea, connecting back to the Vacher and Brusco work. This will be a very powerful conclusion.

----


*   **Chapter 3: Model Development and Methodology**
    *   **Introduction:** A brief paragraph setting the stage for the chapter.
    *   **Section 3.1: Experimental Framework:** This section comes **first**. It sets up the ground rules for everything that follows. It's the "here's how we measure success and what we're working with" section.
        *   `Subsection 3.1.1: Datasets and Rationale` (MIT1003 vs. SALICON).
        *   `Subsection 3.1.2: Evaluation Metrics` (Formal definitions of LL, IG, etc.).
        *   `Subsection 3.1.3: Multi-Stage Training Protocol` (Justification for the 3 phases).
    *   **Section 3.2: Architectural Foundations: The Move from CNNs to ViTs:** This provides the core technical motivation for your first model.
        *   `Subsection 3.2.1: The Limitations of the CNN Paradigm`.
        *   `Subsection 3.2.2: The Advantages of the Vision Transformer Paradigm`.
    *   **Section 3.3: A Narrative of Model Development:** This is the core story of your research innovations.
        *   `Subsection 3.3.1: DinoGaze: Establishing a New ViT-Powered Baseline` (Describes the first model and the ambiguity of its success).
        *    `Subsection 3.3.2: Preparing the Guidance Signal: Unsupervised Segmentation Mask Generation.` **(MOVED HERE).** Now we explain the masks before we use them.
        *   `Subsection 3.3.2: Injecting Explicit Segmentation with SPADE` (Describes the rationale for SPADE and the "semantic incoherence" problem).
        *   `Subsection 3.3.3: The "Semantic Painting" Innovation` (Describes your solution).
    *   **Section 3.4: Final Model Architectures:** This section provides the final, detailed "blueprint" of the models that will be evaluated in the Results chapter.
        *   `Subsection 3.4.1: The DinoGaze Architecture`.
        *   `Subsection 3.4.2: The DinoGaze-SPADE Architecture`.


*   **Chapter 3: Model Development and Methodology (The "What" and "Why")**
    *   This chapter should remain focused on the **conceptual blueprints** of your models. It tells the scientific story of *what* you built and *why* you built it that way. It's about the architectural ideas, the rationale, and the theoretical underpinnings. This is where you convince the reader of the intellectual merit of your approach. The `Final Model Architectures` section absolutely belongs here. It's the climax of your development narrative.

*   **Chapter 4: Implementation and Experimental Setup (The "How")**
    *   Merge your "Tools" and "Implementation" ideas into this single, practical chapter. This is the engineering report. It details *how* you turned the conceptual blueprints from Chapter 3 into a working, reproducible, and scalable research pipeline. This is where you demonstrate rigor and technical competence.
    *   **Content for this chapter would include:**
        *   **Software and Hardware Stack:** Briefly mention PyTorch, `pysaliency`, etc. State the hardware used (e.g., NVIDIA A100 GPUs) and the environment (e.g., a SLURM-managed cluster).
        *   **Training Orchestration:** A high-level description of your `orchestrator.py` script. Explain how it automates the multi-stage, multi-fold training protocol, manages configurations via YAML, and ensures reproducibility.
        *   **Distributed Training:** Explain that you used `torchrun` and `DistributedDataParallel` (DDP) for multi-GPU training, and briefly mention why this is necessary for the scale of your models and datasets.
        *   **Data Pipeline in Detail:** Explain the `dataloader` and `sampler` setup, particularly the custom `ImageDatasetSampler` for training and the challenges of DDP validation sampling.
        *   **Key Code Snippets/Modules (by reference):** You don't need to paste all the code, but you can refer to key files (e.g., "The `DinoV2Backbone` class in `src/dinov2_backbone.py` handles the on-the-fly padding and feature extraction..."). This is where you would also detail the specifics of your mask generation scripts.