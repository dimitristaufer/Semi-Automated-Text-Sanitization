# Semi-Automated Text Sanitization Tool
[![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

The primary motivation behind our tool is to mitigate the risk of whistleblower re-identification.

Read [our paper](https://arxiv.org/abs/2405.01097) for more details. We were accepted for publication at the ACM Conference on Fairness, Accountability, and Transparency 2024 (ACM FAccT'24). The link leads to a preprint manuscript on arXiv.

## Abstract

Whistleblowing is essential for ensuring transparency and accountability in both public and private sectors. However, (potential) whistleblowers often fear or face retaliation, even when reporting anonymously. The specific content of their disclosures and their distinct writing style may re-identify them as the source. Legal measures, such as the EU Whistleblower Directive, are limited in their scope and effectiveness. Therefore, computational methods to prevent re-identification are important complementary tools for encouraging whistleblowers to come forward. However, current text sanitization tools follow a one-size-fits-all approach and take an overly limited view of anonymity. They aim to mitigate identification risk by replacing typical high-risk words (such as person names and other labels of named entities) and combinations thereof with placeholders. Such an approach, however, is inadequate for the whistleblowing scenario since it neglects further re-identification potential in textual features, including the whistleblower's writing style. Therefore, we propose, implement, and evaluate a novel classification and mitigation strategy for rewriting texts that involves the whistleblower in the assessment of the risk and utility. Our prototypical tool semi-automatically evaluates risk at the word/term level and applies risk-adapted anonymization techniques to produce a grammatically disjointed yet appropriately sanitized text. We then use a Large Language Model (LLM) that we fine-tuned for paraphrasing to render this text coherent and style-neutral. We evaluate our tool's effectiveness using court cases from the European Court of Human Rights (ECHR) and excerpts from a real-world whistleblower testimony and measure the protection against authorship attribution attacks and utility loss statistically using the popular IMDb62 movie reviews dataset, which consists of 62 individuals. Our method can significantly reduce authorship attribution accuracy from 98.81% to 31.22%, while preserving up to 73.1% of the original content's semantics, as measured by the established cosine similarity of sentence embeddings.

## Citation

```
@article{staufer2024silencing,
  title={Silencing the Risk, Not the Whistle: A Semi-automated Text Sanitization Tool for Mitigating the Risk of Whistleblower Re-Identification},
  author={Staufer, Dimitri and Pallas, Frank and Berendt, Bettina},
  journal={arXiv preprint arXiv:2405.01097},
  year={2024}
}
```

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: https://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg
