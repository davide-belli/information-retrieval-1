- All the documentation on what the code does and how to run it can be found directly inside the notebook.
- The <model>.run results are created in './rankings/' after the execution of the different tasks.
- The bash script eval.sh can be used to perform 'trec_eval' evaluation on these files.
- At the end of each task evaluation, comparisons and statistics are computed for each model, and most relevant results are included into the report.

The directory './evaluations/' will contain the results of the evaluations using the validation set ('./ap_88_89/qrel_validation')
Instead, the directory './test/' will contain the evaluation with respect to the test set ('./ap_88_89/qrel_test')
