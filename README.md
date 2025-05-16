# Contra4: Evaluating Contrastrive Cross-Modal Reasoning in Audio, Video, Image, and 3D
[🔗 Project Page](https://artemisp.github.io/contra4-web/)


![Diagram](assets//contra4_examples.png)

## Data
We store the data in an easy to use `json` format. 
We release both the train, validation, and test sets in `data/final_data`. Each data point is a JSON object with the following keys:

* `id` (string): A unique identifier for this specific question.
* `selection_type` (string): Indicates the type of question. In this case, it is always `"similarity"`.
* `q_type` (string): Specifies the question subtype. Here, it is `"mc_2"`, denoting a multiple-choice question with two options.
* `examples` (array of objects): An array containing two example scenes or items to be compared. Each element in this array is an object with the following keys:
    * `source` (string): Indicates the origin or source of the example (e.g., `"pointllm"`, `"msrvtt_test"`).
    * `id` (string): A unique identifier for the specific example within its source.
    * `meta` (object, optional): Additional metadata associated with the example. This can vary depending on the source.
    * `url` (string, optional): A URL providing access to the visual or media content of the example (e.g., a Sketchfab embed link or a YouTube video link).
    * `caption` (string): A textual description or caption associated with the example.
* `gt_answer` (string): The ground truth answer to the question. This will typically refer to one of the scenes (e.g., `"Scene A"`, `"Scene B"`).
* `question` (string): The actual question being asked, prompting a comparison between the provided examples.
* `modalities`: (list of strings) The modality type of each example

**Example Data Point:**

```json
{
    "id": "similarity_368568",
    "selection_type": "similarity",
    "q_type": "mc_2",
    "examples": [
        {
            "source": "pointllm",
            "id": "69865c89fc7344be8ed5c1a54dbddc20",
            "meta": {},
            "url": "[https://sketchfab.com/models/69865c89fc7344be8ed5c1a54dbddc20/embed](https://sketchfab.com/models/69865c89fc7344be8ed5c1a54dbddc20/embed)",
            "caption": "A storyed building"
        },
        {
            "source": "msrvtt_test",
            "id": "video6747",
            "caption": "a 16bit game with people in a field near a fence",
            "url": "[https://www.youtube.com/watch?v=2fZswgVxzYE](https://www.youtube.com/watch?v=2fZswgVxzYE)",
            "meta": {
                "start_time": 1137.05,
                "end_time": 1160.68
            }
        }
    ],
    "gt_answer": "Scene B",
    "question": "Which scene involves more physical activity?",
    "modalities": ["pc", "video"]
}
```

## Raw Data Download



## Data Generation

We include the scripts for generating multimodal question-answering datasets. 
---

### **Environment Setup**
- Install Python 3.9 or higher.
- Install required Python packages:
  ```bash
  pip install -r src/data_generation/requirements.txt
  ```

### Step 1: Sampling Data
Run step1_sampling.py to sample multimodal data and generate tuples for the task.

```bash
python src/data_generation/step1_sampling.py --strategy <sampling_strategy> --n_samples <num_samples> --split <split>
```

- **Arguments**:
  - `--strategy`: Sampling strategy (`random` or `similarity`).
  - `--n_samples`: Number of samples to generate per modality combination.
  - `--split`: Dataset split (`train`, `test`, etc.).

- **Output**:
  - JSON file containing sampled tuples saved in step1.

---

### Step 2: Generate Questions and Answers
Run step2_3_question_answer.py to generate questions and answers for the sampled data.

```bash
python src/data_generation/step2_3_question_answer.py --strategy <sampling_strategy> --split <split> --bs <batch_size> --model_id <model_id>
```

- **Arguments**:
  - `--strategy`: Sampling strategy (`random` or `similarity`).
  - `--split`: Dataset split (`train`, `test`, etc.).
  - `--bs`: Batch size for processing.
  - `--model_id`: Model ID for question/answer generation (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

- **Output**:
  - JSON file containing questions and answers saved in step2_3.

---

### Step 3: Run Model Inference for Permutations
Run step4_rtc.py to evaluate the generated questions and answers using a model.

```bash
python src/data_generation/step4_rtc.py --model <model_name> --strategy <sampling_strategy> --split <split> --batch_size <batch_size>
```

- **Arguments**:
  - `--model`: Model name (`llama`, `mistral`, etc.).
  - `--strategy`: Sampling strategy (`random` or `similarity`).
  - `--split`: Dataset split (`train`, `test`, etc.).
  - `--batch_size`: Batch size for processing.

- **Output**:
  - JSON file with model predictions saved in step4.

---

### Step 4: Apply Filters
Run filters.py to filter the generated data based on various criteria.

```bash
python src/data_generation/filters.py
```

- **Arguments**:
  - Modify `MODEL_LIST` and `SELECTION_LIST` in the script to specify models and strategies.

- **Output**:
  - Filtered datasets saved in `data/filters_<split>/`.

---

### Step 5: Balance the Dataset
Run balance.py to balance the dataset across different modalities and question types.

```bash
python src/data_generation/balance.py
```

- **Arguments**:
  - Modify `selection_type` and `split` variables in the script to specify the selection type and dataset split.

- **Output**:
  - Balanced dataset saved in `data/filters_<split>/`.

---

### Step 6: Categorize Questions
Run category.py to categorize the generated questions into predefined categories.

```bash
python src/data_generation/category.py --strategy <sampling_strategy> --split <split> --model_id <model_id>
```

- **Arguments**:
  - `--strategy`: Sampling strategy (`random` or `similarity`).
  - `--split`: Dataset split (`train`, `test`, etc.).
  - `--model_id`: Model ID for categorization (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

- **Output**:
  - Categorized dataset saved in `data/filters_<split>/`.

---

## Notes
- Ensure all required datasets are downloaded and placed in the raw_data directory.
- Modify file paths in the scripts if your directory structure differs.
- Use GPUs for faster execution, especially for encoding and model inference tasks.

---

## Example Workflow
To generate a dataset using the `similarity` strategy for the `test` split:
```bash
python src/data_generation/step1_sampling.py --strategy similarity --n_samples 30000 --split test
python src/data_generation/step2_3_question_answer.py --strategy similarity --split test --bs 16 --model_id meta-llama/Llama-3.1-8B-Instruct
python src/data_generation/step4_rtc.py --model llama --strategy similarity --split test --batch_size 16
python src/data_generation/filters.py
python src/data_generation/balance.py
python src/data_generation/category.py --strategy similarity --split test --model_id meta-llama/Llama-3.1-8B-Instruct
```

---

## Outputs
- **Step 1**: `data/step1/tuples_<strategy>_<split>.json`
- **Step 2 & 3**: `data/step2_3/<strategy>_<split>.json`
- **Step 4**: `data/step4/<model>_<strategy>_<split>.json`
- **Filters**: `data/filters_<split>/`
- **Balanced Dataset**: `data/filters_<split>/unanimous_permute_filter_<strategy>_<split>_balanced.json`
- **Categorized Dataset**: `data/filters_<split>/unanimous_permute_filter_<strategy>_<split>_balanced.json`


## Cross-Modal Baselines

### Annotation Files
For Audio and 3D data the annotation files are automatically downloaded from web urls. For MSRVTT download the train and val parquet files in `raw_data/video/msrvtt/` from here [train](https://huggingface.co/datasets/AlexZigma/msr-vtt/blob/main/data/train-00000-of-00001-60e50ff5fbbd1bb5.parquet) and [val](https://huggingface.co/datasets/AlexZigma/msr-vtt/blob/main/data/val-00000-of-00001-01bacdd7064306bc.parquet). For images, the 2017 train and validation json annotation files can be found here [train](https://huggingface.co/datasets/merve/coco/blob/main/annotations/captions_train2017.json) and [val](https://huggingface.co/datasets/merve/coco/blob/main/annotations/captions_val2017.json).

### Raw Data

**Image Data**
 * [MSCOCO](https://cocodataset.org/#home): Download the 2017 MSCOCO dataset: [train images](http://images.cocodataset.org/zips/train2017.zip) and [val images](http://images.cocodataset.org/zips/val2017.zip)

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
* [Objaverse](https://objaverse.allenai.org/) the formatted data for OneLLM and X-InstructBLIP can be found in `objaverse_pc_parallel` [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research/ULIP-Objaverse_triplets;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forclseOnObjectsSortingFiltering=false). For CREMA, Objaverse data should be preprocessed as described in [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM)

**Video Data**
* [MSRVTT](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_.pdf) can be downloaded from [here](https://mega.nz/file/UnRnyb7A#es4XmqsLxl-B7MP0KAat9VibkH7J_qpKj9NcxLh8aHg). 

#### Environment variables
You should set the following environment variables based on the locations where you have stored your datasets. Modify the paths according to your system:
```
export COCO_DIR="/path/to/your/coco/images"
export AUDIOCAPS_DIR="/path/to/your/audiocaps/audios"
export MSRVTT_DIR="/path/to/your/msrvtt/videos"
export CLOTHO_DIR="/path/to/your/clotho/audios"
export OBJAVERSE_FEAT_DIR="/path/to/your/objaverse/pointclouds"
```

#### Run Cross-Modal Baselines
In `cross_modal_baselines/` clone the corresponding code for each baseline and follow the instructions to create **separate** environments for each of them.  
 * [X-InstructBLIP](https://arxiv.org/abs/2311.18799). To set-up the environment run the commands below.
 ```
 cd cross_modal_baselines/
 git clone https://github.com/salesforce/LAVIS.git
 cd LAVIS
 conda create -n lavis python=3.10
 conda activate lavis
 python -m pip install -e .
 pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
 wget -P /usr/bin https://github.com/unlimblue/KNN_CUDA/raw/master/ninja
 pip install -r LAVIS/requirements.txt
 ```
 To run the baseline run 
 ```
 conda activate lavis
 python xinstructblip.py
 ```
 * [CREMA](https://arxiv.org/abs/2402.05889)/. To setup Crema follow the instructions below. Note that for CREMA, Objaverse data should be preprocessed as described in [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM)

 ```
 git clone https://github.com/Yui010206/CREMA.git
 conda create -n crema python=3.8
 conda activate crema 
 cd CREMA
 pip install -e .
 ```
 To run the baseline run 
 ```
 conda activate crema
 python crema.py
 ```
* [OneLLM](https://arxiv.org/abs/2402.05889)/. To setup Crema follow the instructions below.
 ```
 git clone https://github.com/csuhan/OneLLM
 conda create -n onellm python=3.9 -y
 conda activate onellm
 pip install -r requirements.txt # make sure to update torch version
 export CUDA_HOME=$CONDA_PREFIX
 cd model/lib/pointnet2
 python setup.py install
 ```
  To run the baseline run 
 ```
 conda activate onellm
 python onellm.py
 ```
 * [Gemini]. Install the required libraries
 ```
 pip install google-genai torch tqdm
 ```
Obtain a Google API key for the Gemini model and set it as an environment variable:
```
export GOOGLE_API_KEY="<your_api_key>"
```
To run the baseline run 
 ```
 python gemini.py
 ```
* [Caption Baseline]. Install the required libraries. Generated captions are created using the scripts in `src/caption_models` and are saved in the caption field of each modality example json. 
 ```
 pip install google-genai torch tqdm vllm
 ```
Then run the baseline 
```
python caption_baseline.py --model_id <model_id>
```

#### FineTune OneLLM

The src/cross_modal_baselines/contra4_finetune.py script is used to fine-tune the OneLLM model for cross-modal reasoning tasks. Follow the steps below to set up and run the fine-tuning process.

#### Setup Fine-Tuning Environment

1. **Install Required Libraries**:
   Ensure you have the following libraries installed:
   ```bash
   pip install torch torchvision pytorchvideo tqdm tensorboard
   ```
   Also unzip the zipped files in data/final_data

2. **Prepare Checkpoints**:
   Download the pre-trained OneLLM checkpoints and place them in the appropriate directory (e.g., `OneLLM/OneLLM-7B/`).

#### Run the Fine-Tuning Script

1. **Run the Script**:
   Execute the contra4_finetune.py script to fine-tune the OneLLM model:
   ```bash
   python contra4_finetune.py --batch_size <batch_size> --epochs <num_epochs> --output_dir <output_dir>
   ```

   - **Arguments**:
     - `--batch_size`: Batch size per GPU (default: `2`).
     - `--epochs`: Number of epochs to train (default: `1`).
     - `--output_dir`: Directory to save the fine-tuned model and logs (default: `./output`).
     - `--llama_ckpt_dir`: Path to the pre-trained OneLLM checkpoint directory (default: `OneLLM/OneLLM-7B`).
     - `--llama_config`: Path to the OneLLM configuration file (default: `OneLLM/config/llama2/7B.json`).
     - `--tokenizer_path`: Path to the tokenizer model (default: `OneLLM/config/llama2/tokenizer.model`).

2. **Outputs**:
   - Fine-tuned model checkpoints will be saved in the specified `--output_dir`.
   - Training logs will be saved in the same directory.

## Citation
```
TBA
```