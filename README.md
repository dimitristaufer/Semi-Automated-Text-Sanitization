# Semi-Automated Text Sanitization

The following is the implementation part of my master thesis at Technische Universität Berlin.

![screenshot](<Screenshot-Link-Here>)

## Table of Contents

- [Project Overview](#project-overview)
- [Anonymization Techniques](#anonymization-techniques)
- [Project Evaluation](#project-evaluation)
- [Comparative Advantages](#comparative-advantages)
- [Setup and Installation](#setup-and-installation)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

The primary motivation behind our tool is to mitigate the risk of whistleblower re-identification.

![Project Overview Diagram](<Diagram-Link-Here>)

## Anonymization Techniques

We use a combination of three anonymization operations designed explicitly for unstructured data:

- Suppression: Removes words and dependent phrases that may carry sensitive information.
- Generalization: Leverages knowledge bases like Wikidata and WordNet to replace sensitive terms with more general ones.
- Perturbation: Utilizes a fine-tuned FLAN T5 language model for rephrasing text to obfuscate the author's writing style.

## Project Evaluation

We assessed the effectiveness of our tool through an automatic evaluation. The evaluation metrics included authorship attribution accuracy, semantic similarity (S-BERT embeddings), and utility loss (BERT-based sentiment classifier) on the IMDb62 movie review dataset. The tool reduced the authorship attribution accuracy of state-of-the-art classifiers from 98.81% to 31.22% while preserving up to 73.1% of the original semantics.

## Comparative Advantages

Our tool takes a novel approach to text sanitization that takes into account both the lexical and syntactic dimensions of text. It offers a user-friendly interface that acknowledges the context-dependent nature of sensitivity, offering the user a seamless experience while ensuring utmost confidentiality.

## Setup and Installation

Our tool is provided as a Docker container for ease of deployment. You can find our Docker image on [Docker Hub](<Docker-Hub-Link-Here>). You may also clone the entire source code and our fine-tuned models from this repository. To get started, follow the instructions in our [Setup and Installation guide](<Your-Installation-Guide-Link-Here>).

## Future Work

While our tool presents promising results, we acknowledge certain limitations tied to tokenization, anonymization, excessive removal, and hallucination. We recommend further qualitative evaluation to increase the effectiveness and applicability of our tool. 

## License

This project is licensed under the terms of the MIT license.

---
Project maintained by [Dimitri Staufer](https://www.dimitristaufer.com/contact).
