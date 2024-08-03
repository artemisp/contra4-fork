# Datasheet for dataset "DisCRn"

Questions from the [DisCRn]() paper, v1.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

### For what purpose was the dataset created? 

The dataset was created to address the lack of benchmarks capable of evaluating artificial intelligence systems across multiple modalities simultaneously. It aims to facilitate research in discriminatory cross-modal reasoning, an area that enhances the understanding of complex, multimodal information by integrating audio, video, image, and 3D data.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
Artemis Panagopoulou, Le Xue, Honglu Zhou, Silvio Savarese, Ran Xu, Caiming Xiong, Juan Carlos Niebles in [Salesforce AI Research](https://www.salesforceairesearch.com/) 

### Who funded the creation of the dataset? 
All work was conducted during employment in [Salesforce AI Research](https://www.salesforceairesearch.com/).

## Composition


### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?
The dataset comprises links to multimedia data across different modalities—audio files, 3D models, images, and video clips. Each instance is accompanied by a question and corresponding answer that focus on cross-modal discrimination.


### How many instances are there in total (of each type, if appropriate)?
There are 65,535 examples in the dataset. From them 24,959 are samples using high similarity, and 40,576 using random similarity. There are 51,717 examples with two choices, 13,706 with three, and 112 with four. 

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

The dataset was filtered from an automatically generated dataset of 574,906 samples. We release this original sample as well, and the filtering code. Refer to the [paper]() for more details. 

### What data does each instance consist of? 

Each instance includes multimedia metadata (audio, 3D, image, video) of captioning datasets along with a question related to the content, and the corresponding answer. We release the data in `json` format as follows:
```
{
    "id": "r7",
    "selection_type": "random",
    "q_type": "mc_2",
    "examples": [
        {
            "source": "clothov1_instruct_val",
            "id": "street 2.wav",
            "caption": "A busy street with a car shifting gears in traffic"
            "url": ""
        },
        {
            "source": "objaverse_pointllm_val",
            "id": "760c0d78327b4846975061c6cd8fd004",
            "caption": "a red sports car with black wheels."
            "url": ""
        }
    ],
    "modalities": [
        "audio",
        "pc"
    ],
    "questions": [
        "Which scene  evokes more motion?"
    ],
    "answers": [
        "Scene A"
    ],
    "category": "Motion"
}
```
# Structure
- **id**: Unique identifier for the dataset entry.
- **selection_type**: The method used for selecting negative examples.
- **q_type**: the question type indicating the number of choices.
- **examples**:
  - `source`: The dataset from which the example is taken.
  - `iD`: A unique identifier for the example within its source.
  - `caption`: A description of the content or scene depicted in the example.
  - `url`: The URL to the example if it exists
- **modalities**: the modalities of each of the provided examples. The i'th modality in `modalities` corresponds the modality of the i'th example in `examples`.
- **questions**: Example question. 
- **answers**: Ground truth answer. 
- **Category**: Question category (predicted using in context learning with LLaMa-2 13b)


### Is there a label or target associated with each instance?

There is a question and a multiple choice answer for each instance. 


### Are there recommended data splits (e.g., training, development/validation, testing)?

The dataset is all intended for evaluation purposes.

### Are there any errors, sources of noise, or redundancies in the dataset?

Since it is an automatically generated dataset, there are instances with incorrect responses, estimated to 6.7% of the total examples. 

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

_If it links to or relies on external resources, a) are there guarantees that they will
exist, and remain constant, over time; b) are there official archival versions of the
complete dataset (i.e., including the external resources as they existed at the time the
dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with
any of the external resources that might apply to a future user? Please provide descriptions
of all external resources and any restrictions associated with them, as well as links or other
access points, as appropriate._

## Collection process

### How was the data associated with each instance acquired?

The data is associated with image, 3D, audio, and video captioning data each of which can be downloaded following the instructions below. When available, we directly include the link to the datapoint as described in [Composition](#composition). 

**Image Data**
 * [MSCOCO](https://cocodataset.org/#home): Download the MSCOCO dataset Val2014 from [here](http://images.cocodataset.org/zips/val2014.zip)
 * [Densely Captioned Images](https://github.com/facebookresearch/DCI?tab=readme-ov-file): Download the Densely Captioned Images source from [here](https://scontent.xx.fbcdn.net/m1/v/t6/An_zz_Te0EtVC_cHtUwnyNKODapWXuNNPeBgZn_3XY8yDFzwHrNb-zwN9mYCbAeWUKQooCI9mVbwvzZDZzDUlscRjYxLKsw.tar?ccb=10-5&oh=00_AYAxjfFtxB_hSfxuY6SRYPGueZwDgHjTPetZDgieJdsi7g&oe=6682228A&_nc_sid=0fdd51) after accepting the terms of [SA-1B](https://ai.meta.com/datasets/segment-anything/). 

**Audio Data**

We recommend using [`aac-datasets`](https://github.com/Labbeti/aac-datasets) to download the audio data. 
* [AudioCaps](https://audiocaps.github.io/)
```
from aac_datasets import AudioCaps
dataset = AudioCaps(root="/path/to/save/folder", subset="val", download=True)
```
* [ClothoV1](https://zenodo.org/records/3490684)
```
from aac_datasets import Clotho
dataset = Clotho(root="/path/to/save/folder", subset="eval", download=True)
```
* [ClothoV2](https://zenodo.org/records/4783391)
```
from aac_datasets import Clotho
dataset = Clotho(root="/path/to/save/folder", subset="val", download=True)
```

**3D Data**
* [Objaverse](https://objaverse.allenai.org/) the formatted data for OneLLM and X-InstructBLIP can be found in `objaverse_pc_parallel` [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research/ULIP-Objaverse_triplets;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false). For CREMA, Objaverse data should be preprocessed as described in [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM)


**Video Data**
* [MSRVTT](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_.pdf) can be downloaded from [here](https://mega.nz/file/UnRnyb7A#es4XmqsLxl-B7MP0KAat9VibkH7J_qpKj9NcxLh8aHg). 
* [Charades](https://prior.allenai.org/projects/charades) can be downloaded from [here](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip)

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?
The multimdedia data was sourced from existing datasets as described above, and the data generation was performed using an open sourced language model [LLaMA2-13b](https://huggingface.co/meta-llama/Llama-2-13b) from [Huggingface](https://huggingface.co/).

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

<!-- ### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)? -->

### Over what timeframe was the data collected?
The dataset was created during the first quarter (January to May) of 2024.

### Were any ethical review processes conducted (e.g., by an institutional review board)?
N/A

## Preprocessing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?
We introduce a four step data generation and filtering pipeline described in the [paper](). At a high level it operates as follows: In Step 1, candidate choices are sampled either randomly or by selecting those with high text similarity. Step 2 employs in-context learning to generate a question based on the captions, which is then answered in Step 3. Step 4 utilizes a mixture-of-models round-trip consistency check to eliminate incorrect samples.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?
We release the unfilterd dataset on [GitHub]() in the folder `data/step2_3`.

### Is the software used to preprocess/clean/label the instances available?
All code used to generate the data is available on [GitHub]() in the folder `data_generation`.

## Uses

### Has the dataset been used for any tasks already?
Yes, the dataset has been employed in preliminary studies to benchmark the performance of existing multimodal AI systems, testing their ability to integrate and reason about information from different sensory inputs. See [paper]().


### Are there tasks for which the dataset should not be used?
It should not be used for training models since it relies on evaluation subsets of existing datasets.


## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 
NA
<!-- _If so, please provide a description._ -->

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?
Will be released in a GitHub repository. 

### When will the dataset be distributed?
The dataset will be released following paper review. 


### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?
CC-by-4.0

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?
All datasets used have research friendly licenses as discussed in the [paper]().

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?
All datasets used have research friendly licenses as discussed in the [paper]().

## Maintenance

### Who is supporting/hosting/maintaining the dataset?
The dataset will be maintained by the Salesforce AI Research team, with periodic updates and community input facilitated through [GitHub](). 

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?
The corresponding author is Artemis Panagopoulou (email: [artemisp@seas.upenn.edu](mailto:artemisp@seas.upenn.edu?subject=[GitHub]DisCRn))

### Is there an erratum?
An erratum will be hosted on the [GitHub]() repository, where corrections and updates to the dataset documentation or data itself will be posted as needed.


### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?
Yes, updates will be made to correct any discovered errors, reflect changes in linked resources, or respond to community feedback. Notification of updates will be communicated through the project's [GitHub]() repository.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?
N/A

### Will older versions of the dataset continue to be supported/hosted/maintained?
Older versions of the dataset will be archived and available for download. The version history will be maintained on GitHub, allowing researchers to access previous versions for comparative studies or replication purposes.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?
Contributions can be processed through GitHub pull requests. 