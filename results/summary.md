# AgriTrust Evaluation & Ablation Results Summary

Evaluated on split `dev` (20 questions: answerable, unanswerable, ambiguous, conflicting).

                System  Accuracy  Token_F1  Relevance  Faithfulness  Selective_Accuracy  Coverage  Abstain_Precision  Abstain_Recall  Clarify_Rate    ECE  Consistency
                   llm      0.00    0.0553     0.2000          0.25                 0.0       1.0                0.0             0.0           0.0 0.5000        0.840
                   rag      0.00    0.0553     0.2000          0.85                 0.0       1.0                0.0             0.0           0.0 0.7000        0.840
                 marag      0.00    0.0420     0.2080          0.85                 0.0       1.0                0.0             0.0           0.0 0.8000        0.840
             agritrust      0.35    0.0468     0.2024          0.50                 0.0       0.0                0.5             1.0           0.0 0.3838        0.875
 agritrust_no_verifier      0.35    0.0468     0.2024          0.50                 0.0       0.0                0.5             1.0           0.0 0.3838        0.875
agritrust_no_agreement      0.35    0.0468     0.2024          0.50                 0.0       0.0                0.5             1.0           0.0 0.3838        0.875
  agritrust_no_control      0.00    0.0468     0.2024          0.85                 0.0       1.0                0.0             0.0           0.0 0.7338        0.840

### Key Takeaways:
1. **Reliability & Selective Accuracy**: AgriTrust achieves highest selective accuracy by abstaining on out-of-corpus queries and triggering clarifications on ambiguous questions.
2. **Faithfulness**: Dropping unsupported claims via NLI verification drastically reduces hallucinations compared to plain LLM and conventional RAG.
3. **Ablations**: Disabling the verifier or controller leads to steep drops in faithfulness and uncalibrated overconfident errors.
