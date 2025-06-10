# From History to FUTURE and Back Again: Bidirectional Fuzzing for Deep Learning Libraries

This is the source code repo for "**From History to FUTURE and Back Again: Bidirectional Fuzzing for Deep Learning Libraries**".

## Introduction
FUTURE is the first universal fuzzing framework tailored for both newly introduced and prospective DL libraries. This framework significantly reduces the effort required to adapt fuzzing techniques to any new DL library, making it a forward-thinking tool. Additionally, FUTURE is the first fuzzing method that targets Apple MLX. FUTURE establishes a code conversion mapping between existing and prospective libraries by fine-tuning LLMs. This mapping allows it to convert historical bug codes from existing libraries into seed codes for prospective libraries, pioneering a fuzzing method from historical insights to future anticipations. FUTURE demonstrates remarkable efficacy by detecting 155 bugs in Apple MLX, Huawei MindSpore, and OneFlow, including 143 previously unknown bugs. Among these bugs, 10 have been assigned CVE IDs. In addition, FUTURE has identified 10 bugs in PyTorch and 12 bugs in TensorFlow. Beyond bug detection, FUTURE identifies 20 enhancement issues and 12 documentation problems, further showcasing its effectiveness and adaptability.

Here is the overview of FUTURE: 
![OVERVIEW](https://github.com/Redmept1on/FUTURE_v2/blob/main/overview.png)

## Getting Started

### Requirements

The experiments are conducted on an Ubuntu 20.04 server equipped with an Intel Xeon Gold 6130 CPU (6 cores) and a V100-32GB GPU. 

Our basic environment: `PyTorch v2.0.0, TensorFlow v2.13.0, Python 3.8, Cuda 11.8`

- MLX v0.0.10 install following https://github.com/ml-explore/mlx?tab=readme-ov-file#installation
- MindSpore v2.2.13 install following https://www.mindspore.cn/install/en
- OneFlow v0.9.1 install following https://github.com/Oneflow-Inc/oneflow

### Step 1 - Environment configuration
Clone the repo and we recommend using the same environment as mentioned above.

### Step 2 - Run
We organize the codes according to the three main components of the FUTURE and we provide instructions for each component. You can find detailed README in each folder for more details.

#### 1. Historical bug collection
Remember to adjust "folder_name" in `spider.py` according to your own local path and you can see the README in the corresponding folder for detailed instructions.

```
cd 1_historical_bug_collection
python crawler.py
```

#### 2. Seed Code Generation
Follow the instructions in the corresponding folder for detailed instructions.
The sequence of running the code is as follows: 
api info extraction -> code pair generation -> mutate code pair -> equivalence evaluation -> fine-tuning🔥 -> merge model

#### 3. Generation and Conversion
After obtaining the fine-tuned models, we use them to perform code conversion between libraries and code generation.

You can follow `./3_gen_and_conversion/convert.ipynb` and `./3_gen_and_conversion/randomgen.ipynb` to perform conversion and generation with the fine-tuned CodeLlama respectively. Remember to change the "model_id" to the catalogue of your merged model.

#### 4. Feedback to Source Libraries
Follow  `./4_feedback_to_source_libraries/README.md` to extract bug-prone API input values from bug reports discovered by FUTURE using fine-tuned models.

## Reproduce Bugs in Paper
All the bugs FUTURE detected are in `./detected_bugs/reproduction`. You can access the statistic pics of bugs by running `python ./detected_bugs/statistic.py`.
