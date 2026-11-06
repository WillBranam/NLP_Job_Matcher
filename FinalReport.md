FINAL PROJECT AND REPORT

The project will compare a base small LLM with a LoRA fine-tuned version for matching recent graduates’ tech résumés to entry-level jobs and internships. The system should rank postings, assign a 0–100 matching score, and explain matching qualifications and missing requirements.

The final report should describe the datasets, preprocessing, model, LoRA training setup, and evaluation methods. It should compare matching accuracy, keyword overlap, explanation quality, and computational cost, then discuss errors and limitations. An API model comparison is an additional planned experiment within the budget.

WHAT TO IMPLEMENT FOR THE MIDTERM

1. Download and inspect the résumé–job training dataset and job-posting dataset.
2. Prepare and save separate training and testing data, documenting filtering, sample counts, and duplicate handling.
3. Implement loading the proposed small LLM and running inference on a résumé–job pair.
4. Implement LoRA adapter setup and the training pipeline, including input formatting, tokenization, and training configuration.
5. Demonstrate that a short training run executes successfully. Save the adapter and verify that it reloads for inference.

WHAT TO INCLUDE IN THE MIDTERM REPORT

Briefly describe the project, downloaded datasets, prepared splits, model, and LoRA implementation. Include actual execution evidence, such as dataset summaries, model output, and training logs. Clearly state any unfinished or unverified work.

The instructor does not require performance results or proof of improvement at the midterm. The requirement is to show that the datasets are available and the model and LoRA training implementation can run.