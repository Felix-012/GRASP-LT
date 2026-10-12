# Prompt embeddings at generation time

Generation reads one precomputed prompt embedding per row of the test CSV (`--precomputed_embeddings_path`,
indexed by row). The registry records, per configuration, the file used for the reported run (`emb_gen`) and
the file of the test prompts (`emb_gen_fixed`).

For the label-conditioned configurations `ft_class`, `ft_text_class`, `ccua` and `vanilla_mg`, the reported
runs read the *training-split* embedding file (`embeddings_mimic-lt_labels_train.h5`, resp. `..._label_txt_train.h5`)
while iterating over the test CSV, so the condition of test row *j* is the label of training row *j*.
The run files reproduce this by default. `--fix_gen_embeddings` (CLI) or `EXTRA=--fix_gen_embeddings`
(run files) generates from the test-prompt embeddings instead (`embeddings_mimic-lt_labels_test.h5`; for
`ft_text_class` build `embeddings_mimic-lt_label_txt_test.h5` first).

All other configurations (GRASP, MoE-GRASP, MoCLE, Adapter, the variants, ImageNet-LT) read the test-prompt
embeddings in both modes.
