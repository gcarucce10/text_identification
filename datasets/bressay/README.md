# BRESSAY: Brazilian Essays Dataset for Handwritten Text Recognition

The BRESSAY dataset comprises images of handwritten essays in Brazilian Portuguese, which present a series of challenges to optical recognition models. These images were sourced from multiple online platforms, limiting our ability to standardize the capture process. Due to these varied sources and the lack of a uniform collection method, the dataset provides a realistic reflection of real-world conditions. Each essay is unique, contributed by different writers, and addresses a specific content topic. Furthermore, the constraints placed on the writers often lead to various handwriting scenarios, including hard-to-read words, connected words, noise, overwriting, and struck-through texts.

## Dataset Composition

The BRESSAY dataset represents a comprehensive collection of handwritten essays in Brazilian Portuguese, offering detailed insights into various handwriting scenarios. It covers a total of 1,000 pages, each contributed by a unique writer, resulting in 1,000 distinct handwriting styles. This aspect of the dataset adds a layer of diversity, which is further emphasized by the total of 4,214 paragraphs, 30,090 lines, and 416,826 words. Regarding unique tokens, we have 41,318 unique words, and 107 unique characters.

## Data Structure

The dataset is organized as follows:

- `data/`: Main folder containing segmented essay images
  - `lines/`: Images of individual lines
    - PNG files: Line images
    - TXT files: Transcriptions of lines
  - `pages/`: Full page essay images
    - PNG files: Page images
    - TXT files: Transcriptions of pages
  - `paragraphs/`: Images of paragraphs
    - PNG files: Paragraph images
    - TXT files: Transcriptions of paragraphs
  - `words/`: Images of individual words
    - PNG files: Word images
    - TXT files: Transcriptions of words

- `sets/`: Contains partition files
  - `test.txt`: Names of images in the test set
  - `validation.txt`: Names of images in the validation set
  - `training.txt`: Names of images in the training set

## Dataset Usage and Annotations

Each name in test.txt, validation.txt and training.txt represents the name of the page and all its content (words, lines, paragraphs) must be in the respective partition.

Annotations used in the dataset:
  - `##@@???@@##`: Unidentifiable superscript text. Superscript text that has become unidentifiable and unreadable.
  - `$$@@???@@$$`: Unidentifiable subscript text. Subscript text that has become unidentifiable and unreadable.
  - `@@???@@`: Unidentifiable text. Text that cannot be read or identified due to its illegibility.
  - `##--xxx--##`: Superscript and illegibly crossed out. Text that has been added as a superscript and subsequently crossed out, rendering it illegible.
  - `$$--xxx--$$`: Subscript and illegibly crossed out. Text that has been added as a subscript and subsequently crossed out, rendering it illegible.
  - `--xxx--`: Unreadable strikethrough. Text that has been crossed out in a way that makes it unreadable.
  - `##--text--##`: Superscript and legibly crossed out. Text that has been added as a superscript and subsequently crossed out, but remains legible.
  - `$$--text--$$`: Subscript and legibly crossed out. Text that has been added as a subscript and subsequently crossed out, but remains legible.
  - `##text##`: Superscript text in the line. Text added as a superscript in the line, typically as a correction or additional note.
  - `$$text$$`: Subscript text in the line. Text added as a subscript in the line, typically as a correction or additional note.
  - `--text--`: Readable strikethrough. Text that has been crossed out but remains readable.

## Terms of Usage

The BRESSAY dataset may be used for non-commercial research and teaching purposes only. If you are publishing scientific work based on the BRESSAY dataset, we request you to include a reference to our paper:

- Neto, A.F.S., Bezerra, B.L.D., Araujo, S.S., Souza, W.M.A.S., Alves, K.F., Oliveira, M.F., Lins, S.V.S., Hazin, H.J.F., Rocha, P.H.V., Toselli, A.H.: BRESSAY: A Brazilian Portuguese Dataset for Offline Handwritten Text Recognition. In: 18th International Conference on Document Analysis and Recognition (ICDAR). Springer, Athens, Greece (9 2024).

- Neto, A.F.S., Bezerra, B.L.D., Araujo, S.S., Souza, W.M.A.S., Alves, K.F., Oliveira, M.F., Lins, S.V.S., Hazin, H.J.F., Rocha, P.H.V., Toselli, A.H.: ICDAR 2024 Competition on Handwritten Text Recognition in Brazilian Essays – BRESSAY. In: 18th International Conference on Document Analysis and Recognition (ICDAR). Springer, Athens, Greece (9 2024).
