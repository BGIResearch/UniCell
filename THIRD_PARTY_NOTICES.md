# Third-party notices

The MIT license for UniCell's original contributions does not replace the
licenses of incorporated code or data. The complete distribution is not
MIT-only: it includes GPL v3 code, Apache-2.0 code, MIT/BSD code, and CC BY 4.0
ontology data. Preserve the applicable licenses, copyright notices, and source
attributions when redistributing these components.

The revisions below were checked on 2026-09-27 as source and license references.
They are **not claims about the exact original import commits**; those commits
were not recorded. Some files contain preexisting UniCell adaptations, in
addition to English documentation translations. Existing source headers and
acknowledgements are retained.

## Main incorporated components

| Local scope | Upstream reference and license | Included license |
|---|---|---|
| `unicell/repo/geneformer/` | [Geneformer historical reference](https://huggingface.co/ctheodoris/Geneformer/tree/b2bbd7ccc856e3b3c1a9199c7cca07d888d87663); the [model card](https://huggingface.co/ctheodoris/Geneformer/blob/b2bbd7ccc856e3b3c1a9199c7cca07d888d87663/README.md) declares Apache-2.0. Author: Christina Theodoris. | `licenses/Apache-2.0.txt` |
| `unicell/repo/scgpt/`; borrowed functions in `scfoundation/genemodule/plot_geneemb.ipynb` | [scGPT](https://github.com/bowang-lab/scGPT/tree/cebd6fae655b9c585a4807daa3ac31bb764f06b4), MIT. Copyright (c) 2022 suber. | `licenses/scGPT-MIT.txt` |
| `unicell/repo/scfoundation/`, subject to the incorporated-code exceptions below | [scFoundation](https://github.com/biomap-research/scFoundation/tree/948a8ccb950d096148cf03418d870acdcadebd7b), Apache-2.0. Copyright 2023 BioMap (Beijing) Intelligence Technology Limited. | `licenses/scFoundation-Apache-2.0.txt`; original `unicell/repo/scfoundation/LICENSE` |
| `unicell/repo/scbert/`; scBERT-derived Performer/reversible implementations in scFoundation | [scBERT](https://github.com/TencentAILabHealthcare/scBERT/tree/262fd4b91f3f1c21a6e595d03d4ef423e16ffc99), GPL v3. Copyright (c) 2023 Tencent Inc. | `licenses/scBERT-GPL-3.0.txt` |
| `unicell/utils/onclass_utils.py` | [OnClass](https://github.com/wangshenguiuc/OnClass/tree/8910b07876542e819c279f5a4e53801f0eb27a86), MIT. Copyright (c) 2021 sheng wang. The local `read_data` function accepts an AnnData input and preserves its sparse expression matrix. | `licenses/OnClass-MIT.txt` |
| `unicell/cl-basic.obo`; its derived `unicell/graph.gml` | [Cell Ontology release 2024-02-13](https://github.com/obophenotype/cell-ontology/tree/c4d8797979da6b1a20899151a5659568fabeac51), CC BY 4.0, credited to the Cell Ontology contributors. The OBO header identifies this release and license. The GML file is an ontology-derived graph used by UniCell. | `licenses/Cell-Ontology-CC-BY-4.0.txt` |

scBERT-derived code includes `performer.py` and `reversible.py` under
`unicell/repo/scfoundation/pretrainmodels/` and
`unicell/repo/scfoundation/model/pretrainmodels/`, and related implementations
under `unicell/repo/scfoundation/GEARS/modules/`. The surrounding scFoundation
Apache license does not replace the terms of these incorporated sources.
An MIT grant for original UniCell contributions does not waive GPL requirements
applicable to distribution of a combined GPL-covered work, including providing
its corresponding source under the applicable terms.

## Additional incorporated sources

The local paths below are relative to `unicell/repo/`.

| Component and local use | Upstream reference | Included license and attribution |
|---|---|---|
| DeepCDR, `scfoundation/DeepCDR/` | [DeepCDR](https://github.com/kimmo1019/DeepCDR/tree/4dc5a901d580511335b9a54ffce9fb188f9f068d) | `licenses/DeepCDR-MIT.txt`; Copyright (c) 2019 Qiao Liu. |
| GEARS, `scfoundation/GEARS/gears/` | [GEARS](https://github.com/snap-stanford/GEARS/tree/f374e43e197b295016d80395d7a54ddb81cc6769) | `licenses/GEARS-MIT.txt`; Copyright (c) 2022 Yusuf Roohani, Kexin Huang, Jure Leskovec. |
| Performer PyTorch, Performer implementations in scBERT/scFoundation | [performer-pytorch](https://github.com/lucidrains/performer-pytorch/tree/fc8b78441b1e27eb5d9b01fc738a8772cee07127) | `licenses/performer-pytorch-MIT.txt`; Copyright (c) 2020 Phil Wang. This does not relicense later scBERT changes. |
| RevTorch, credited by reversible implementations | [RevTorch](https://github.com/RobinBruegger/RevTorch/tree/bff49ecf092c73729ab240935d902ac211ebf376) | `licenses/RevTorch-BSD-3-Clause.txt`; Copyright (c) 2019, Robin Brügger. |
| MMF, `scfoundation/GEARS/modules/attention.py` | [MMF](https://github.com/facebookresearch/mmf/tree/b0ffc7fba39f210f833c127362146aec01a2f259) | `licenses/MMF-BSD-3-Clause.txt`; Copyright (c) Facebook, Inc. and its affiliates. All rights reserved. |
| Nilearn, version helper attribution in `scfoundation/GEARS/gears/version.py` | [Nilearn 0.9.0](https://github.com/nilearn/nilearn/tree/72e810f01ddd05aa28573edf9aedbaad6ddc2a98) | `licenses/Nilearn-BSD-3-Clause.txt`; Copyright (c) 2007 - 2015 The nilearn developers. Existing Loic Esteve/Ben Cipollini attribution is retained. |
| Google Performer, fast-attention source references | [historical implementation](https://github.com/google-research/google-research/blob/097fc5829f351faac3117760f19a52571c43a373/performer/fast_attention/jax/fast_attention.py); [license reference](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/LICENSE) | `licenses/Google-Performer-Apache-2.0.txt`; Copyright 2021 The Google Research Authors. |
| genevector, `scgpt/tasks/grn.py` | [genevector](https://github.com/nceglia/genevector/tree/bbb7c5dcdf200cc5321ef17dfe67209f97aa2208) | `licenses/genevector-MIT.txt`; Copyright (c) 2020 Nick Ceglia. |
| SimCSE, contrastive embedding gathering in scGPT model files | [SimCSE](https://github.com/princeton-nlp/SimCSE/tree/49fa580a853752ede55b8c76d9debf748e214d3f) | `licenses/SimCSE-MIT.txt`; Copyright (c) 2021 Princeton Natural Language Processing. |
| PyTorch, adapted samplers/encoder layers and checkpoint references | [PyTorch v1.13.1 license](https://github.com/pytorch/pytorch/blob/v1.13.1/LICENSE) | `licenses/PyTorch-BSD.txt`; full upstream copyright and license notices retained. |
| torchtext, vocabulary construction in `scgpt/tokenizer/gene_tokenizer.py` | [torchtext v0.14.1](https://github.com/pytorch/text/tree/v0.14.1) | `licenses/torchtext-BSD-3-Clause.txt`; Copyright (c) James Bradbury and Soumith Chintala 2016. |
| Hugging Face Transformers, Geneformer collator/trainer/tokenization helpers | [Transformers v4.26.1](https://github.com/huggingface/transformers/tree/v4.26.1) | `licenses/Transformers-Apache-2.0.txt`; Copyright 2020 The HuggingFace Team. All rights reserved. Copyright 2020 The HuggingFace Inc. team. |

The reference tags identify checked license evidence; they do not determine
the versions of separately installed runtime dependencies.

## Unresolved upstream documentation

- **SCAD:** `unicell/repo/scfoundation/SCAD/` contains an adaptation of
  [CompBioT/SCAD](https://github.com/CompBioT/SCAD/tree/c9cea4cefc55593fde7e92330ac24db579df1001).
  No code redistribution grant was found in the checked repository, license
  history, README, or issue discussions at
  [the current reference](https://github.com/CompBioT/SCAD/tree/fa145688a989e20b45f7db9d8a502b96ab2807a2).
  The paper's public code link is not a software license. This is an unresolved
  permission-documentation gap, not evidence of an express prohibition.
  This notice does not supply the missing permission.
- **DSBN:** `unicell/repo/scgpt/model/dsbn.py` is an unchanged copy, apart from
  line endings, of the checked MIT-licensed scGPT file. It credits
  [the original DSBN source](https://github.com/woozch/DSBN/blob/6edb3199bae6fb001e1bde0dcd548d52599a985c/model/dsbn.py).
  No separate license was found in that original repository. No DSBN exclusion
  was found in scGPT's MIT declaration. The immediate scGPT grant and original
  attribution are retained; the earlier authorization documentation remains
  unverified.

## Model weights and external data

These source-code notices do not grant rights to separately downloaded model
weights or datasets. The bundled scFoundation model README matches the March
2024 reference that describes weights as CC BY-NC-SA 4.0; later upstream
releases use a distinct noncommercial license. Retain the terms and provenance
of the specific assets used. No external model weight is relicensed as MIT by
this repository's original-code license.

The license copies in `licenses/` and this notice accompany source distributions,
application wheels, and newly assembled offline bundles. Previously built
release artifacts are not retroactively changed by an update to these sources.
