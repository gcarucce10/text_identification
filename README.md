# Library for AI-based Handwritten Text Identification in Documents 📝🤖

This repository contains the project for the "Projetos em Computação I". The project focuses on developing a library using Artificial Intelligence to identify and transcribe handwritten text from document images.

---

## 🎯 Project Overview

The core idea is to create an AI-powered library capable of transcribing handwritten text found in images of documents. While the concept might seem straightforward, it involves several developmental challenges, ranging from modeling and training an AI to making it available for practical use in end-user tools.

This project was proposed by the students: Filipi Peruca de Melo Pereira, Gabriel Freitas Caruccę, and Mariana Ferreira Kenworthy. It is a new project, not a continuation of previous work, and is not associated with any extension projects.

### 🌟 Objectives
* To create a library based on Artificial Intelligence with the functionality to transcribe written elements from images of text documents.
* Explore diverse applications, such as assisting people with visual impairments or serving as a document digitalization tool.
* Acknowledge the potential for future expansion regarding user interfaces, including websites, applications, or other digital platforms.

### 🌱 Sustainable Development Goals (ODS)
This project aligns with the following UN Sustainable Development Goals:
* Quality Education 
* Industry, Innovation, and Infrastructure 

---

## 🧑‍💻 Team

### Students
| Name                          | UNESP ID  | Email                      | Responsibilities (Initial)                                                                 |
| :---------------------------- | :-------- | :------------------------- | :----------------------------------------------------------------------------------------- |
| Filipi Peruca de Melo Pereira | 221151176 | filipi.pm.pereira@unesp.br | Planning, Modeling, Development & Testing, Final Adjustments, Final Report |
| Gabriel Freitas Carucce       | 221153871 | gabriel.carucce@unesp.br   | Planning, Development & Testing, Extra Deliverable Development, Final Adjustments, Final Report |
| Mariana Ferreira Kenworthy    | 221150986 | m.kenworthy@unesp.br       | Planning, Modeling, Extra Deliverable Development, Final Adjustments, Final Report (lead) |

### 👨‍🏫 Supervisor
* **Name**: Daniel Carlos Guimarães Pedronette 
* **Department**: DEMAC 

### ✨ Postgraduate Collaborators
| Name                       | Level | Program         | Email                    |
| :------------------------- | :---- | :-------------- | :----------------------- |
| Gustavo Rosseto Leticio    | D     | PPGCC-UNESP     | gustavo.leticio@unesp.br |
| Vinicius Atsushi Sato Kawai | D     | PPGCC-UNESP     | vinicius.kawai@unesp.br  |

---

## 🛠️ Technology Stack
* **Programming Language**: Python 
* **IDE**: Visual Studio Code 
* **Core Libraries**: Pytorch/TensorFlow + Keras 

---

## 💡 Areas of Application
* Artificial Intelligence 
* Computer Vision
* Machine Learning 
* Mobile 
* Web 

---

##  Deliverables
* A final report detailing the project's development and results.
* A prototype of the library (Desktop) featuring the core methods developed, along with their descriptions.
* **Optional**: An implementation for an alternative platform (initially Web or Mobile), contingent on project progress.
* Archived final codes for the main deliverable and the extra deliverable (if applicable), along with documentation.
* Documentation and versioning of tools used for future reproducibility.

---

##  metodology Methodology & Approach

### Initial Steps:
1.  **Modeling** (Filipi, Mariana):
    * Selection of models with the highest implementation viability from the bibliography.
    * Initial implementation of models using PyTorch/TensorFlow + Keras.
    * Definition of a basic/rudimentary pipeline to obtain initial results.
2.  **Development and Testing** (Filipi, Gabriel):
    * Training models with selected datasets.
    * Adjustments to the pipeline and its stages to achieve acceptable accuracy.
    * Saving pre-trained models for later use.

### Approach:
The initial idea is to implement a model based on **CNN + RNN or transformers**[cite: 28]. This will follow a pre-processing stage for the images, which may include:
* DPI adjustment 
* Deskewing (cisalhamento) 
* Segmentation 
* Resolution adjustment 

### 📊 Datasets
The following datasets are being considered for training, validation, and testing:
* **IAM Dataset**: An English sentence database for offline handwriting recognition. Available at: [https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database) 
* **BRESSAY Dataset**: A Brazilian Portuguese Dataset for Offline Handwritten Text Recognition. Available at: [https://tc11.cvc.uab.es/datasets/BRESSAY_1](https://tc11.cvc.uab.es/datasets/BRESSAY_1)

---

## 🗓️ Project Timeline (1st Semester 2025)

The project activities are planned across several weeks, from late May to the end of June.

* **Planejamento (Planning)** (All team members): Periodic meetings for discussion and planning of project progress.
* **Modelagem (Modeling)** (Filipi, Mariana): Selection and initial implementation of models.
* **Desenvolvimento e Testagem (Development and Testing)** (Filipi, Gabriel): Training models and refining the pipeline.
* **Desenvolvimento de Entregável Extra (Extra Deliverable Development)** (Gabriel, Mariana): If project progress is satisfactory, development of an additional deliverable using the pre-trained models.
* **Ajustes Finais (Final Adjustments)** (All team members): Bug fixing, final tweaks, and beginning documentation.
* **Relatório Final (Final Report)** (All team members, with Mariana leading): Archiving code, documentation, and producing the final report detailing project progress and results.

---

## 📚 Initial Bibliography

1.  Neto, A.F.S., Bezerra, B.L.D., Araujo, S.S., Souza, W.M.A.S., Alves, K.F., Oliveira, M.F., Lins, S.V.S., Hazin, H.J.F., Rocha, P.H.V., Toselli, A.H.: BRESSAY: A Brazilian Portuguese Dataset for Offline Handwritten Text Recognition. In: 18th International Conference on Document Analysis and Recognition (ICDAR). Springer, Athens, Greece (2024). 
2.  Marti, UV., Bunke, H. The IAM-database: an English sentence database for offline handwriting recognition. IJDAR 5, 39-46 (2002). 
3.  A. F. de Sousa Neto, B. L. D. Bezerra, A. H. Toselli and E. B. Lima, "HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition" 2020 33rd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), Porto de Galinhas, Brazil, 2020, pp. 54-61. 
4.  T. Bluche and R. Messina, "Gated Convolutional Recurrent Neural Networks for Multilingual Handwriting Recognition," 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), Kyoto, Japan, 2017, pp. 646-651. 
5.  M. Fujitake, "DTrOCR: Decoder-only Transformer for Optical Character Recognition," 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Waikoloa, HI, USA, 2024, pp. 8010-8020. 
6.  Kang, L., Riba, P., Wang, Y., Rusiñol, M., Fornés, A., Villegas, M. (2020). GANwriting: Content-Conditioned Generation of Styled Handwritten Word Images. In: Vedaldi, A., Bischof, H., Brox, T., Frahm, JM. (eds) Computer Vision – ECCV 2020. ECCV 2020. Lecture Notes in Computer Science(), vol 12368. Springer, Cham. 

---
